import unittest
from unittest.mock import Mock
from online_discovery import youtube_music_search_url, verified_spotify_match


class SpotifyVerificationTests(unittest.TestCase):
    def test_unicode_youtube_link(self):
        url = youtube_music_search_url("Rzira", "2月25日")
        self.assertIn("music.youtube.com/search", url)
        self.assertIn("%E6%9C%88", url)

    def test_exact_artist_and_title(self):
        client = Mock()
        client.search.return_value = {"tracks": {"items": [
            {"name": "Another Song", "artists": [{"name": "Rzira"}],
             "external_urls": {"spotify": "https://open.spotify.com/track/wrong"}},
            {"name": "2月25日", "artists": [{"name": "Rzira"}],
             "external_urls": {"spotify": "https://open.spotify.com/track/right"}},
        ]}}
        self.assertEqual(verified_spotify_match("Rzira", "2月25日", client),
                         "https://open.spotify.com/track/right")

    def test_artist_mismatch_not_verified(self):
        client = Mock()
        client.search.return_value = {"tracks": {"items": [
            {"name": "Same Track", "artists": [{"name": "Different Artist"}],
             "external_urls": {"spotify": "https://open.spotify.com/track/wrong"}},
        ]}}
        self.assertIsNone(verified_spotify_match("My Artist", "Same Track", client))


if __name__ == "__main__":
    unittest.main()
