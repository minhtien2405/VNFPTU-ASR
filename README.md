# VNFPTU-ASR

## Installation

```
    git clone https://github.com/minhtien2405/VNFPTU-ASR.git
    cd https://github.com/minhtien2405/VNFPTU-ASR.git
    uv venv
    source .venv/bin/activate
    uv pip sync requirements.lock
    pre-commit install
```

create a .env file in the root directory with the following content:

```
    OPENAI_API_KEY=your_openai_api_key
    GOOGLE_APPLICATION_CREDENTIALS=path_to_your_google_credentials.json
    ...
```
