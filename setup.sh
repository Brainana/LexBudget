mkdir -p ~/.streamlit/

echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml

open_ai_api_key=`heroku config:get OPENAI_API_KEY`
sed -i "s/your-api-key-here/$open_ai_api_key/g" ./.streamlit/secrets.toml

assistant_id=`heroku config:get ASSISTANT_ID`
sed -i "s/your-assistant-id-here/$assisant_id/g" ./config.ini


