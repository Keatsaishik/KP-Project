# Define the mapping as a dictionary
status = {
    500: "Success",
    400: "OpenAI API request timed out",
    401: "OpenAI API returned an API Error",
    402: "OpenAI API request failed to connect",
    403: "OpenAI API request was invalid",
    404: "OpenAI API request was not authorized",
    405: "OpenAI API request was not permitted",
    406: "OpenAI API request exceeded rate limit"
}

