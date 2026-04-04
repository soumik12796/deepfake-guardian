import { GoogleGenAI, GenerateContentResponse } from "@google/genai";

// ✅ Correct API key usage
const API_KEY = import.meta.env.VITE_API_KEY;

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
        error?.status === "RESOURCE_EXHAUSTED" ||
        error?.code === 429;

      if (isQuotaError && i < maxRetries - 1) {
        console.warn(`Retrying in ${delay}ms...`);
        await new Promise((res) => setTimeout(res, delay));
        delay *= 2;
        continue;
      }
      throw error;
    }
  }

  throw new Error("Max retries exceeded");
}

// 📊 Types
export interface AnalysisResult {
  isReal: boolean;
  confidence: number;
  source: string;
  reasoning: string;
  metadata: {
    resolution?: string;
    format?: string;
    artifacts?: string[];
  };
  heatmaps?: Record<string, string>;
}

export type HeatmapMode = "artifacts" | "noise" | "lighting";

// 🔥 Heatmap
export async function generateHeatmap(
  base64Image: string,
  mode: HeatmapMode = "artifacts"
): Promise<string> {
  const model = "gemini-1.5-flash"; // ✅ safer model

  const prompts = {
    artifacts: "Highlight deepfake artifacts as heatmap.",
    noise: "Show noise inconsistency heatmap.",
    lighting: "Show lighting inconsistency heatmap",
  };

  const response: GenerateContentResponse = await withRetry(() =>
    ai.models.generateContent({
      model,
      contents: [
        {
          role: "user",
          parts: [
            { text: prompts[mode] },
            {
              inlineData: {
                data: base64Image.split(",")[1],
                mimeType: "image/jpeg",
              },
            },
          ],
        },
      ],
    })
  );

  return response.text || "";
}

// 🔍 Analysis
export async function analyzeImage(
  base64Image: string
): Promise<AnalysisResult> {
  const model = "gemini-1.5-pro"; // ✅ stable model

  const prompt = `
Analyze this image and return JSON:
{
  "isReal": boolean,
  "confidence": number,
  "source": "string",
  "reasoning": "string",
  "metadata": {
    "artifacts": ["string"]
  }
}
`;

  const response: GenerateContentResponse = await withRetry(() =>
    ai.models.generateContent({
      model,
      contents: [
        {
          role: "user",
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
      ],
    })
  );

  try {
    return JSON.parse(response.text || "{}");
  } catch (e) {
    console.error("Parse error:", e);
    throw new Error("Analysis failed");
  }
}