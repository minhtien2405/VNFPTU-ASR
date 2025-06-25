# VNFPTU-ASR

## Installation

```
    git clone [[project_url](https://github.com/minhtien2405/VNFPTU-ASR.git)]
    cd [[project_name](https://github.com/minhtien2405/VNFPTU-ASR.git)]
    uv venv
    source .venv/bin/activate
    uv pip sync requirements.lock
    pre-commit install
```

create a .env file in the root directory with the following content:

```
    # .env
    OPENAI_API_KEY=your_openai_api_key
    GOOGLE_APPLICATION_CREDENTIALS=path_to_your_google_credentials.json
    ...
```
