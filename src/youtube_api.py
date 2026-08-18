import logging
from typing import Dict, List, Optional
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from .config import YOUTUBE_API_KEY

logger = logging.getLogger(__name__)

class YouTubeAPI:
    def __init__(self, api_key: str = YOUTUBE_API_KEY):
        if not api_key:
            raise ValueError(
                "YouTube API key is missing. Please set YOUTUBE_API_KEY in the .env file."
            )
        self.api_key = api_key
        try:
            self.youtube = build("youtube", "v3", developerKey=self.api_key)
        except Exception as e:
            logger.error(f"Failed to build YouTube service: {e}")
            raise

    def search_videos(
        self, query: str, max_results: int = 50, page_token: Optional[str] = None
    ) -> Dict:
        """Search for videos based on a query."""
        try:
            request = self.youtube.search().list(
                part="id,snippet",
                q=query,
                type="video",
                maxResults=max_results,
                pageToken=page_token,
                order="relevance",
            )
            response = request.execute()
            return response
        except HttpError as e:
            logger.error(f"HTTP Error {e.resp.status} occurred during search: {e.content}")
            raise

    def get_video_details(self, video_ids: List[str]) -> Dict:
        """Get details for a list of video IDs (max 50 per request)."""
        if not video_ids:
            return {}
        try:
            request = self.youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=",".join(video_ids)
            )
            response = request.execute()
            return response
        except HttpError as e:
            logger.error(f"HTTP Error {e.resp.status} occurred getting video details: {e.content}")
            raise

    def get_channel_details(self, channel_ids: List[str]) -> Dict:
        """Get details for a list of channel IDs (max 50 per request)."""
        if not channel_ids:
            return {}
        try:
            request = self.youtube.channels().list(
                part="snippet,statistics",
                id=",".join(channel_ids)
            )
            response = request.execute()
            return response
        except HttpError as e:
            logger.error(f"HTTP Error {e.resp.status} occurred getting channel details: {e.content}")
            raise
