import { GoogleGenAI, GenerateContentResponse } from "@google/genai";

// ✅ API KEY
const API_KEY = import.meta.env.VITE_API_KEY;

if (!API_KEY) {
  throw new Error("API key missing! Check VITE_API_KEY in .env");
}

const ai = new GoogleGenAI({
  apiKey: API_KEY,
});

// 🔁 Retry logic
async function withRetry<T>(fn: () => Promise<T>, maxRetries = 3): Promise<T> {
  let delay = 1000;

  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error: any) {
      const isQuotaError =
        error?.message?.includes("RESOURCE_EXHAUSTED") ||
        error?.code === 429;

      if (isQuotaError && i < maxRetries - 1) {
        console.warn(`Retrying in ${delay}ms...`);
        await new Promise((res) => setTimeout(res, delay));
        delay *= 2;
        continue;
      }

      console.error("API ERROR:", error);
      throw error;
    }
  }

  throw new Error("Max retries exceeded");
}

// 📊 TYPE
export interface AnalysisResult {
  isReal: boolean;
  confidence: number;
  source: string;
  reasoning: string;
  metadata: {
    artifacts?: string[];
  };
}

// 🔍 MAIN FUNCTION
export async function analyzeImage(
  base64Image: string
): Promise<AnalysisResult> {
  const model = "gemini-1.5-flash";

  const prompt = `
Analyze this image and return ONLY valid JSON:

{
  "isReal": true or false,
  "confidence": number (0-1),
  "source": "AI or Real",
  "reasoning": "short explanation",
  "metadata": {
    "artifacts": ["list of issues"]
  }
}
`;

  const response: GenerateContentResponse = await withRetry(() =>
    ai.models.generateContent({
      model,
      contents: {
        parts: [
          { text: prompt },
          {
            inlineData: {
              data: base64Image.split(",")[1],
              mimeType: "image/jpeg",
            },
          },
        ],
      },
    })
  );

  const rawText = response.text || "";
  console.log("Gemini RAW:", rawText);

  // 🔥 Clean response
  const cleanText = rawText
    .replace(/```json/g, "")
    .replace(/```/g, "")
    .trim();

  try {
    return JSON.parse(cleanText);
  } catch (e) {
    console.error("JSON Parse Failed:", cleanText);
    throw new Error("Analysis failed");
  }
}

// 🔥 OPTIONAL HEATMAP (SAFE VERSION)
export async function generateHeatmap(
  base64Image: string
): Promise<string> {
  const model = "gemini-1.5-flash";

  const response: GenerateContentResponse = await withRetry(() =>
    ai.models.generateContent({
      model,
      contents: {
        parts: [
          { text: "Highlight manipulated regions in this image." },
          {
            inlineData: {
              data: base64Image.split(",")[1],
              mimeType: "image/jpeg",
            },
          },
        ],
      },
    })
  );

  return response.text || "No heatmap available";
}