TRUSTED_CRYPTO = {"BTC", "ETH", "SOL", "USDT", "USDC", "BNB", "XRP"}
REGULATORY_KEYWORDS = {"task force", "commission", "council", "authority", "agency", "committee", "office"}


# File paths
CRYPTO_LIST_PATH = "api_lists/full_crypto_list_coinmarketcap.json"
EXCHANGE_LIST_PATH = "api_lists/exchange_list.json"
DEX_LIST_PATH = "api_lists/dex_list.json"

DEFAULT_LOG_PATH = "logs/coindesk_news_new.json"

# TF-IDF parameters
TFIDF_TOP_N = 10

# Sentiment chunking
DEFAULT_CHUNK_SIZE = 500

# Topic modeling
LDA_TOPICS = 1
LDA_TOP_WORDS = 10

# VADER thresholds
VADER_POS_THRESHOLD = 0.05
VADER_NEG_THRESHOLD = -0.05

# Entity disambiguation confidence threshold
CRYPTO_DISAMBIGUATION_THRESHOLD = 0.4

# RSS Configuration
RSS_URL = "https://www.coindesk.com/arc/outboundfeeds/rss/"

# Request Settings
HTTP_TIMEOUT_SECONDS = 5

