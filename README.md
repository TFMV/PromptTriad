# PromptTriad

**PromptTriad** is a Go API hosted on Cloud Run that leverages three competing AI models (OpenAI, Gemini, and Cohere) to collaboratively engineer and optimize the best possible prompt from any given input.

![PromptTriad](assets/triad.webp)

## Project Status

🚧 This project is currently under construction. 🚧

## Major Goals/Milestones

- [x] Integrate OpenAI, Gemini, and Cohere APIs
- [x] Implement response evaluation and selection logic
- [x] Deploy on Google Cloud Run
- [ ] Add authentication and rate limiting
- [ ] Implement comprehensive logging and monitoring

## How to Use

1. Clone the repository
2. Set the required environment variables (you will need API keys)
3. Run the API server
4. Send a POST request to `/engineer-prompt` with your prompt input

## Example Request

```bash
curl -X POST http://localhost:8080/engineer-prompt -H "Content-Type: application/json" -d '{"input":"Your prompt here"}'
```

## Example Response

```json
{
  "openai_response": "Response from OpenAI",
  "gemini_response": "Response from Gemini",
  "Cohere_response": "Response from Cohere",
  "best_response": "Selected best response"
}
```
