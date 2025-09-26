import instaloader
from urllib.parse import urlparse
from engine.processors.image_processor import ImageProcessor
from app.core.config import settings


def get_instagram_media_urls(post_url):
    loader = instaloader.Instaloader(download_pictures=False,
                                     download_videos=False,
                                     download_video_thumbnails=False,
                                     download_comments=False,
                                     save_metadata=False)

    try:
        # Extract shortcode from URL
        parsed_url = urlparse(post_url)
        path_parts = parsed_url.path.strip('/').split('/')

        shortcode = None
        if 'p' in path_parts:
            shortcode = path_parts[path_parts.index('p') + 1]
        elif 'reel' in path_parts:
            shortcode = path_parts[path_parts.index('reel') + 1]
        elif 'reels/audio' in path_parts:
            shortcode = path_parts[path_parts.index('audio') + 1]
        else:
            print("Invalid Instagram URL.")
            return None

        post = instaloader.Post.from_shortcode(loader.context, shortcode)

        media_urls = {"images": [], "videos": [], "audio": None}

        if post.is_video:
            media_urls["videos"].append(post.video_url)
            # Instagram does not expose a direct audio-only URL
            # Must be extracted with ffmpeg after downloading
        else:
            media_urls["images"].append(post.url)

        # Handle sidecar (multiple media in one post)
        if post.typename == "GraphSidecar":
            for node in post.get_sidecar_nodes():
                if node.is_video:
                    media_urls["videos"].append(node.video_url)
                else:
                    media_urls["images"].append(node.display_url)

        return media_urls

    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    url = "https://www.instagram.com/reel/DMDPPJSuZR4/?utm_source=ig_web_copy_link"#"https://www.instagram.com/reel/DOY2a8BkoYx/?utm_source=ig_web_copy_link&igsh=cG1zZ3JsMzc5bzA5" #"https://www.instagram.com/p/DOqbj1sjEjk/?utm_source=ig_web_copy_link&igsh=MTAwaGt4Y3U0eDdreA=="  
    media = get_instagram_media_urls(url)
    print(media)
    if media:
        print(media['images'][0])
        image_processor = ImageProcessor(settings.API_NINJAS_KEY)
        text = image_processor.process(media['images'][0])
        print(text)