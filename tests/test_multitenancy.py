"""Tests for multitenancy support."""

import pytest

from app.auth import get_stored_token, _token_store, DEFAULT_CLIENT_KEY
from app.oura_client import OuraClient, DataSource, NotAuthenticatedError


class TestTokenStoreMultitenancy:
    """Tests for per-client token storage."""

    def setup_method(self):
        """Clear token store before each test."""
        _token_store.clear()

    def teardown_method(self):
        """Clear token store after each test."""
        _token_store.clear()

    def test_get_stored_token_returns_none_for_unknown_client(self):
        """Test that get_stored_token returns None for unknown client keys."""
        assert get_stored_token("unknown_client") is None

    def test_get_stored_token_returns_token_for_known_client(self):
        """Test that get_stored_token returns token for known client keys."""
        _token_store["client_a"] = {"access_token": "token_a", "refresh_token": "refresh_a"}
        _token_store["client_b"] = {"access_token": "token_b", "refresh_token": "refresh_b"}

        assert get_stored_token("client_a") == "token_a"
        assert get_stored_token("client_b") == "token_b"

    def test_get_stored_token_default_client_key(self):
        """Test that get_stored_token uses default client key when not specified."""
        _token_store[DEFAULT_CLIENT_KEY] = {"access_token": "default_token", "refresh_token": "refresh"}

        assert get_stored_token() == "default_token"

    def test_multiple_clients_isolated(self):
        """Test that multiple clients have isolated token storage."""
        _token_store["user1"] = {"access_token": "token1", "refresh_token": "refresh1"}
        _token_store["user2"] = {"access_token": "token2", "refresh_token": "refresh2"}
        _token_store["user3"] = {"access_token": "token3", "refresh_token": "refresh3"}

        # Each client should get their own token
        assert get_stored_token("user1") == "token1"
        assert get_stored_token("user2") == "token2"
        assert get_stored_token("user3") == "token3"

        # Modifying one client's token shouldn't affect others
        _token_store["user1"]["access_token"] = "new_token1"
        assert get_stored_token("user1") == "new_token1"
        assert get_stored_token("user2") == "token2"
        assert get_stored_token("user3") == "token3"


class TestOuraClientMultitenancy:
    """Tests for OuraClient multitenancy support."""

    def test_oura_client_passes_client_key_to_token_getter(self):
        """Test that OuraClient passes client_key to token_getter."""
        received_keys: list[str] = []

        def mock_token_getter(client_key: str) -> str | None:
            received_keys.append(client_key)
            return f"token_for_{client_key}"

        client = OuraClient(data_source=DataSource.USER, token_getter=mock_token_getter)

        # Call _get_token_or_raise with different client keys
        token1 = client._get_token_or_raise("user_a")
        token2 = client._get_token_or_raise("user_b")
        token3 = client._get_token_or_raise("user_c")

        assert token1 == "token_for_user_a"
        assert token2 == "token_for_user_b"
        assert token3 == "token_for_user_c"
        assert received_keys == ["user_a", "user_b", "user_c"]

    def test_oura_client_raises_when_token_getter_returns_none(self):
        """Test that OuraClient raises NotAuthenticatedError when token is None."""
        def mock_token_getter(client_key: str) -> str | None:
            if client_key == "authenticated_user":
                return "valid_token"
            return None

        client = OuraClient(data_source=DataSource.USER, token_getter=mock_token_getter)

        # Should work for authenticated user
        token = client._get_token_or_raise("authenticated_user")
        assert token == "valid_token"

        # Should raise for unauthenticated user
        with pytest.raises(NotAuthenticatedError):
            client._get_token_or_raise("unauthenticated_user")

    def test_oura_client_sandbox_mode_ignores_client_key(self):
        """Test that sandbox mode doesn't require token_getter and ignores client_key."""
        client = OuraClient(data_source=DataSource.SANDBOX)

        # Should return "sandbox" token regardless of client_key
        assert client._get_token_or_raise("any_client") == "sandbox"
        assert client._get_token_or_raise("another_client") == "sandbox"

    def test_oura_client_single_instance_multiple_users(self):
        """Test that a single OuraClient instance can serve multiple users."""
        tokens = {
            "user1": "token1",
            "user2": "token2",
            "user3": "token3",
        }

        def mock_token_getter(client_key: str) -> str | None:
            return tokens.get(client_key)

        # Single client instance
        client = OuraClient(data_source=DataSource.USER, token_getter=mock_token_getter)

        # Can get tokens for different users from same instance
        assert client._get_token_or_raise("user1") == "token1"
        assert client._get_token_or_raise("user2") == "token2"
        assert client._get_token_or_raise("user3") == "token3"

        # Unknown user should raise
        with pytest.raises(NotAuthenticatedError):
            client._get_token_or_raise("unknown_user")
