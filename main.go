package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"strings"

	cohere "github.com/cohere-ai/cohere-go"
	"github.com/gin-gonic/gin"
	"github.com/google/generative-ai-go/genai"
	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/prompts"
	"google.golang.org/api/iterator"
	"google.golang.org/api/option"
)

type PromptRequest struct {
	Input string `json:"input"`
}

type PromptResponse struct {
	OpenAIResponse string `json:"openai_response"`
	GeminiResponse string `json:"gemini_response"`
	CohereResponse string `json:"cohere_response"`
	BestResponse   string `json:"best_response"`
}

func main() {
	router := gin.Default()

	router.POST("/engineer-prompt", func(c *gin.Context) {
		var request PromptRequest
		if err := c.BindJSON(&request); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
			return
		}

		openAIResponse, err := getOpenAIResponse(request.Input)
		if err != nil {
			log.Printf("OpenAI error: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to get response from OpenAI"})
			return
		}

		geminiResponse, err := getGeminiResponse(request.Input)
		if err != nil {
			log.Printf("Gemini error: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to get response from Gemini"})
			return
		}

		cohereResponse, err := getCohereResponse(request.Input)
		if err != nil {
			log.Printf("Cohere error: %v", err)
			c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to get response from Cohere"})
			return
		}

		bestResponse := selectBestResponse(openAIResponse, geminiResponse, cohereResponse)

		response := PromptResponse{
			OpenAIResponse: openAIResponse,
			GeminiResponse: geminiResponse,
			CohereResponse: cohereResponse,
			BestResponse:   bestResponse,
		}

		c.JSON(http.StatusOK, response)
	})

	router.Run(":8080")
}

func getOpenAIResponse(prompt string) (string, error) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		return "", fmt.Errorf("missing the OpenAI API key, set it in the OPENAI_API_KEY environment variable")
	}

	os.Setenv("OPENAI_API_KEY", apiKey)

	llm, err := openai.New(openai.WithModel("gpt-4"))
	if err != nil {
		return "", fmt.Errorf("failed to create OpenAI client: %w", err)
	}

	promptTemplate := prompts.NewPromptTemplate(
		"Engineer the specified prompt for better performance. {{.text}}",
		[]string{"text"},
	)
	llmChain := chains.NewLLMChain(llm, promptTemplate)

	ctx := context.Background()
	outputValues, err := chains.Call(ctx, llmChain, map[string]any{
		"text": prompt,
	})
	if err != nil {
		return "", fmt.Errorf("failed to call LLM chain: %w", err)
	}

	out, ok := outputValues[llmChain.OutputKey].(string)
	if !ok {
		return "", fmt.Errorf("invalid chain return")
	}

	return out, nil
}

func getGeminiResponse(prompt string) (string, error) {
	ctx := context.Background()
	client, err := genai.NewClient(ctx, option.WithAPIKey(os.Getenv("GEMINI_API_KEY")))
	if err != nil {
		return "", fmt.Errorf("failed to create Gemini client: %w", err)
	}
	defer client.Close()

	model := client.GenerativeModel("gemini-1.0-pro")

	iter := model.GenerateContentStream(ctx, genai.Text(prompt))
	var result strings.Builder
	for {
		resp, err := iter.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			return "", fmt.Errorf("Gemini error: %w", err)
		}
		for _, cand := range resp.Candidates {
			if cand.Content != nil {
				for _, part := range cand.Content.Parts {
					result.WriteString(fmt.Sprintf("%v", part))
				}
			}
		}
		result.WriteString("---")
	}

	return result.String(), nil
}

func getCohereResponse(prompt string) (string, error) {
	apiKey := os.Getenv("COHERE_API_KEY")
	client, err := cohere.CreateClient(apiKey)
	if err != nil {
		return "", fmt.Errorf("failed to create Cohere client: %w", err)
	}

	generation, err := client.Generate(cohere.GenerateOptions{
		Prompt: prompt,
		Model:  "xlarge",
	})
	if err != nil {
		return "", fmt.Errorf("Cohere error: %w", err)
	}

	return generation.Generations[0].Text, nil
}

// Compute cosine similarity between two texts
func cosineSimilarity(text1, text2 string) float64 {
	vec1 := textToVector(text1)
	vec2 := textToVector(text2)
	return dotProduct(vec1, vec2) / (magnitude(vec1) * magnitude(vec2))
}

func textToVector(text string) map[string]float64 {
	vector := make(map[string]float64)
	words := strings.Fields(text)
	for _, word := range words {
		vector[word]++
	}
	return vector
}

func dotProduct(vec1, vec2 map[string]float64) float64 {
	var result float64
	for key, value := range vec1 {
		if value2, found := vec2[key]; found {
			result += value * value2
		}
	}
	return result
}

func magnitude(vec map[string]float64) float64 {
	var result float64
	for _, value := range vec {
		result += value * value
	}
	return math.Sqrt(result)
}

func selectBestResponse(responses ...string) string {
	if len(responses) != 3 {
		return ""
	}

	similarityOA := cosineSimilarity(responses[0], responses[1]) + cosineSimilarity(responses[0], responses[2])
	similarityGemini := cosineSimilarity(responses[1], responses[0]) + cosineSimilarity(responses[1], responses[2])
	similarityCohere := cosineSimilarity(responses[2], responses[0]) + cosineSimilarity(responses[2], responses[1])

	if similarityOA >= similarityGemini && similarityOA >= similarityCohere {
		return responses[0]
	} else if similarityGemini >= similarityOA && similarityGemini >= similarityCohere {
		return responses[1]
	} else {
		return responses[2]
	}
}
