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

sed -i "s/your-api-key-here/$OPENAI_API_KEY/" ./.streamlit/secrets.toml
sed -i "s/your-trubrics-email-here/$TRUBRICS_EMAIL/" ./.streamlit/secrets.toml
sed -i "s/your-trubrics-pswd-here/$TRUBRICS_PASSWORD/" ./.streamlit/secrets.toml

sed -i "s/your-assistant-id-here/$ASSISTANT_ID/" ./config.ini





