import { GoogleGenAI, GenerateContentResponse } from "@google/genai";

export type HeatmapMode = 'artifacts' | 'noise' | 'lighting';

const API_KEY = import.meta.env.VITE_API_KEY;

if (!API_KEY) {
  throw new Error("API key missing! Check VITE_API_KEY in .env");
}

const ai = new GoogleGenAI({ apiKey: API_KEY });

async function withRetry<T>(fn: () => Promise<T>, maxRetries = 3): Promise<T> {
  let delay = 1000;
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error: any) {
      const isQuotaError =
        error?.message?.includes("RESOURCE_EXHAUSTED") || error?.code === 429;
      if (isQuotaError && i < maxRetries - 1) {
        await new Promise((res) => setTimeout(res, delay));
        delay *= 2;
        continue;
      }
      throw error;
    }
  }
  throw new Error("Max retries exceeded");
}

export interface AnalysisResult {
  isReal: boolean;
  confidence: number;
  source: string;
  reasoning: string;
  metadata: {
    artifacts?: string[];
  };
  heatmaps?: Partial<Record<HeatmapMode, string>>;
}

export async function analyzeImage(base64Image: string): Promise<AnalysisResult> {
  const model = "gemini-1.5-flash";

  const prompt = `
Analyze this image and return ONLY valid JSON with no extra text:

{
  "isReal": true or false,
  "confidence": number between 0 and 1,
  "source": "AI Generated" or "Real Photo",
  "reasoning": "brief explanation under 50 words",
  "metadata": {
    "artifacts": ["list of detected issues, or empty array if real"]
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
  const cleanText = rawText.replace(/```json/g, "").replace(/```/g, "").trim();

  try {
    return JSON.parse(cleanText);
  } catch (e) {
    console.error("JSON Parse Failed:", cleanText);
    throw new Error("Analysis failed");
  }
}

export async function generateHeatmap(
  base64Image: string,
  mode: HeatmapMode
): Promise<string> {
  const model = "gemini-1.5-flash";

  const modePrompts: Record<HeatmapMode, string> = {
    artifacts: "Describe the visual artifacts and manipulated regions in this image.",
    noise: "Describe the noise patterns and inconsistencies in this image.",
    lighting: "Describe the lighting inconsistencies and shadows in this image.",
  };

  const response: GenerateContentResponse = await withRetry(() =>
    ai.models.generateContent({
      model,
      contents: {
        parts: [
          { text: modePrompts[mode] },
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

  return response.text || "No analysis available";
}