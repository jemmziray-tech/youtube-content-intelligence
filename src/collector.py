import json
import logging
import pandas as pd
from pathlib import Path
from typing import List, Dict, Set
from datetime import datetime

from .youtube_api import YouTubeAPI
from .config import SEARCH_QUERIES, RAW_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DataCollector:
    def __init__(self):
        self.api = YouTubeAPI()
        self.raw_data_path = RAW_DATA_DIR / "youtube_raw.csv"

    def fetch_videos_for_query(self, query: str, target_count: int) -> List[Dict]:
        """Fetch video IDs and basic snippets for a specific query."""
        logger.info(f"Searching for '{query}'...")
        videos = []
        next_page_token = None

        while len(videos) < target_count:
            # Calculate how many we still need (max 50 per request)
            remaining = target_count - len(videos)
            max_results = min(50, remaining)

            try:
                response = self.api.search_videos(
                    query=query, max_results=max_results, page_token=next_page_token
                )
                
                for item in response.get("items", []):
                    if item["id"]["kind"] == "youtube#video":
                        videos.append(item)
                
                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break
                    
            except Exception as e:
                logger.error(f"Error during search: {e}")
                break

        return videos[:target_count]

    def _extract_video_data(self, video_response: Dict, channels_data: Dict) -> List[Dict]:
        """Extract relevant fields from the API responses."""
        extracted_data = []
        
        for item in video_response.get("items", []):
            video_id = item["id"]
            snippet = item.get("snippet", {})
            content_details = item.get("contentDetails", {})
            statistics = item.get("statistics", {})
            
            channel_id = snippet.get("channelId")
            channel_info = channels_data.get(channel_id, {})
            
            video_data = {
                "video_id": video_id,
                "title": snippet.get("title"),
                "description": snippet.get("description"),
                "published_at": snippet.get("publishedAt"),
                "channel_id": channel_id,
                "channel_title": snippet.get("channelTitle"),
                "category_id": snippet.get("categoryId"),
                "duration": content_details.get("duration"),
                "tags": ",".join(snippet.get("tags", [])),
                "default_language": snippet.get("defaultLanguage") or snippet.get("defaultAudioLanguage"),
                
                # Video Statistics
                "views": statistics.get("viewCount"),
                "likes": statistics.get("likeCount"),
                "comments": statistics.get("commentCount"),
                
                # Channel Statistics
                "subscriber_count": channel_info.get("subscriberCount"),
                "channel_video_count": channel_info.get("videoCount"),
                "channel_view_count": channel_info.get("viewCount"),
                
                # Metadata
                "collection_date": datetime.utcnow().isoformat()
            }
            extracted_data.append(video_data)
            
        return extracted_data

    def collect(self, target_per_query: int = 10, queries: List[str] = None):
        """Main collection workflow."""
        if queries is None:
            queries = SEARCH_QUERIES
            
        all_video_ids: Set[str] = set()
        all_videos_to_process = []
        
        # 1. Load existing data to avoid re-fetching details unnecessarily if we just want to append.
        # But for statistics updates, we *should* re-fetch details for all known IDs.
        # For this version, we will fetch new details and then merge.
        existing_df = pd.DataFrame()
        if self.raw_data_path.exists():
            existing_df = pd.read_csv(self.raw_data_path)
            all_video_ids.update(existing_df["video_id"].tolist())
            logger.info(f"Loaded {len(existing_df)} existing videos from {self.raw_data_path}")

        # 2. Search for videos across all queries
        new_video_items = []
        for query in queries:
            items = self.fetch_videos_for_query(query, target_count=target_per_query)
            for item in items:
                v_id = item["id"]["videoId"]
                if v_id not in all_video_ids:
                    all_video_ids.add(v_id)
                    new_video_items.append(v_id)

        logger.info(f"Found {len(new_video_items)} new unique videos across queries.")

        if not new_video_items:
            logger.info("No new videos to fetch details for.")
            return

        # 3. Fetch Video Details in batches of 50
        processed_data = []
        for i in range(0, len(new_video_items), 50):
            batch_ids = new_video_items[i:i + 50]
            logger.info(f"Fetching video details batch {i//50 + 1}/{(len(new_video_items)-1)//50 + 1}")
            
            try:
                video_response = self.api.get_video_details(batch_ids)
                
                # 4. Fetch Channel Details for this batch
                channel_ids = list(set([
                    item["snippet"]["channelId"] 
                    for item in video_response.get("items", []) 
                    if "snippet" in item and "channelId" in item["snippet"]
                ]))
                
                channels_data = {}
                for j in range(0, len(channel_ids), 50):
                    channel_batch = channel_ids[j:j + 50]
                    channel_response = self.api.get_channel_details(channel_batch)
                    for c_item in channel_response.get("items", []):
                        channels_data[c_item["id"]] = c_item.get("statistics", {})
                
                # 5. Extract and combine
                extracted = self._extract_video_data(video_response, channels_data)
                processed_data.extend(extracted)
                
            except Exception as e:
                logger.error(f"Error fetching details for batch: {e}")

        # 6. Save and Merge
        if processed_data:
            new_df = pd.DataFrame(processed_data)
            
            if not existing_df.empty:
                # Combine. In a continuous setup, we might also want to update old video stats.
                # For now, we append new videos.
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                # Drop duplicates just in case
                combined_df.drop_duplicates(subset=["video_id"], keep="last", inplace=True)
            else:
                combined_df = new_df
                
            combined_df.to_csv(self.raw_data_path, index=False)
            logger.info(f"Saved data to {self.raw_data_path}. Total videos: {len(combined_df)}")

if __name__ == "__main__":
    collector = DataCollector()
    collector.collect(target_per_query=120) # 120 * 9 queries = 1080 videos
