Good — here's a practical rundown of solid free (or free-tier) APIs, grouped by what people usually build with them. "Free" here means either fully free or a free tier generous enough for a small/hobby project. Free tiers do change, so double-check limits before you commit.

**AI / LLMs**
- Google Gemini API — has an actual free tier (rate-limited) rather than just trial credits.
- Groq — free API access to open models (Llama, Mixtral) with very fast inference.
- Hugging Face Inference API — free tier for many open models.
- OpenRouter — routes to several models; some are free.
- Cohere — free trial tier for text/embeddings.

(Anthropic's own Claude API doesn't have a permanent free tier — it's pay-as-you-go with initial credits, so I'd skip it if "free forever" is a hard requirement.)

**General data / utility**
- OpenWeatherMap — weather, free tier.
- REST Countries — country data, fully free, no key.
- Open-Meteo — weather/climate, free, no key.
- NASA APIs — imagery, astronomy, free with a key.
- The Cat/Dog APIs, JSONPlaceholder — great for prototyping/testing.

**Maps / location**
- OpenStreetMap / Nominatim — geocoding, free (respect usage limits).
- Mapbox — free tier for maps and geocoding.

**Finance / crypto**
- CoinGecko — crypto prices, free tier, no key for basics.
- Alpha Vantage — stock/forex data, free key with rate limits.

**Auth / infra**
- Supabase, Firebase — backend-as-a-service with free tiers (auth, DB, storage).

The right pick depends heavily on what you're building. If you tell me the project — what it does and roughly the scale — I can point you to the two or three that actually fit and flag which free-tier limits would bite you first.
