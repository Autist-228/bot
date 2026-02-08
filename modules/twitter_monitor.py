import asyncio
import httpx
import logging

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import TWITTER_API_KEY, TWITTER_API_BASE

logger = logging.getLogger(__name__)

HEADERS = {
    "X-API-Key": TWITTER_API_KEY,
    "Content-Type": "application/json",
}


async def search_token_mentions(ticker: str) -> dict:
    url = f"{TWITTER_API_BASE}/twitter/tweet/advanced_search"
    query = f"${ticker} OR #{ticker}"
    params = {
        "query": query,
        "queryType": "Latest",
        "cursor": "",
    }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            for attempt in range(3):
                resp = await client.get(url, params=params, headers=HEADERS)
                if resp.status_code == 429:
                    wait = 2 ** attempt + 1
                    logger.warning("Twitter rate limited for %s, waiting %ds", ticker, wait)
                    await asyncio.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            else:
                logger.warning("Twitter rate limit exhausted for %s", ticker)
                return _empty_result(ticker)
            data = resp.json()

        tweets = data.get("tweets", [])
        if not tweets:
            return {
                "ticker": ticker,
                "tweet_count": 0,
                "total_likes": 0,
                "total_retweets": 0,
                "total_views": 0,
                "unique_authors": 0,
                "top_tweet": None,
                "influencer_mentions": 0,
                "avg_followers": 0,
            }

        total_likes = 0
        total_retweets = 0
        total_views = 0
        authors = set()
        influencer_count = 0
        follower_counts = []
        top_tweet = None
        top_engagement = 0

        for tweet in tweets:
            likes = tweet.get("likeCount", 0) or 0
            retweets = tweet.get("retweetCount", 0) or 0
            views = tweet.get("viewCount", 0) or 0
            total_likes += likes
            total_retweets += retweets
            total_views += views

            author = tweet.get("author", {})
            author_id = author.get("id", "")
            authors.add(author_id)
            followers = author.get("followers", 0) or 0
            follower_counts.append(followers)

            if followers >= 10000:
                influencer_count += 1

            engagement = likes + retweets * 2 + views * 0.01
            if engagement > top_engagement:
                top_engagement = engagement
                top_tweet = {
                    "text": tweet.get("text", "")[:200],
                    "author": author.get("userName", ""),
                    "followers": followers,
                    "likes": likes,
                    "retweets": retweets,
                    "views": views,
                    "url": tweet.get("url", ""),
                }

        avg_followers = (sum(follower_counts) / len(follower_counts)
                         if follower_counts else 0)

        return {
            "ticker": ticker,
            "tweet_count": len(tweets),
            "total_likes": total_likes,
            "total_retweets": total_retweets,
            "total_views": total_views,
            "unique_authors": len(authors),
            "top_tweet": top_tweet,
            "influencer_mentions": influencer_count,
            "avg_followers": round(avg_followers),
        }
    except Exception as e:
        logger.error("Twitter search error for %s: %s", ticker, e)
        return {
            "ticker": ticker,
            "tweet_count": 0,
            "total_likes": 0,
            "total_retweets": 0,
            "total_views": 0,
            "unique_authors": 0,
            "top_tweet": None,
            "influencer_mentions": 0,
            "avg_followers": 0,
            "error": str(e),
        }


def _empty_result(ticker: str) -> dict:
    return {
        "ticker": ticker,
        "tweet_count": 0,
        "total_likes": 0,
        "total_retweets": 0,
        "total_views": 0,
        "unique_authors": 0,
        "top_tweet": None,
        "influencer_mentions": 0,
        "avg_followers": 0,
    }


def compute_social_score(mention_data: dict) -> int:
    score = 0

    tweet_count = mention_data.get("tweet_count", 0)
    if tweet_count >= 20:
        score += 2
    elif tweet_count >= 5:
        score += 1

    if mention_data.get("influencer_mentions", 0) >= 1:
        score += 2

    total_engagement = (
        mention_data.get("total_likes", 0)
        + mention_data.get("total_retweets", 0) * 2
    )
    if total_engagement >= 500:
        score += 2
    elif total_engagement >= 100:
        score += 1

    if mention_data.get("total_views", 0) >= 100000:
        score += 1

    if mention_data.get("unique_authors", 0) >= 10:
        score += 1

    return min(score, 8)
