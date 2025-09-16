# Function to fetch transcript from a given Youtube URL

from youtube_transcript_api import YouTubeTranscriptApi

def fetch_youtube_transcript(video_id: str, lang: str = "en") -> str:
    '''Fetches the transcript of a YouTube video given its video ID and transcript language.'''

    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(
        video_id,
        languages=[lang])  # the language of the transcript, default is English. Should try to translate it to English at a later stage if transcript is not available in English

    # Converting the fetched transcript into a single string
    transcript = " ".join([snippet.text for snippet in fetched_transcript])
    return transcript