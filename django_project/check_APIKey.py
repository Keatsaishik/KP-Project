import openai

def is_api_key_valid(API_KEY):

    openai.api_key = API_KEY

    try:
        response = openai.Completion.create(
            engine="davinci",
            prompt="This is a test.",
            max_tokens=5
        )
    except:
        return False
    else:
        return True

