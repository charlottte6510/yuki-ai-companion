# Leave these blank for the commercial build -- every person who runs this
# app pastes their OWN free Groq/Fish Audio key into the in-app Settings
# panel instead (see load_settings()/update_api_keys() in server.py). That's
# what makes distributing this free for you: their usage counts against
# their own free-tier quota, not yours, and you're never redistributing
# access to a key that isn't theirs.
#
# These two only ever act as a fallback default, used purely if someone
# hasn't entered their own key in Settings yet -- so keep them blank here.
#
# Get a free Groq key at https://console.groq.com -> API Keys
GROQ_API_KEY = ""

# Get a free Fish Audio key at https://fish.audio/app/api-keys/
FISH_API_KEY = ""
