import { GoogleGenAI, GenerateContentResponse } from "@google/genai";

const apiKey = process.env.GEMINI_API_KEY || "";
const ai = new GoogleGenAI({ apiKey });

async function withRetry<T>(fn: () => Promise<T>, maxRetries = 3): Promise<T> {
  let delay = 1000;
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error: any) {
      const isQuotaError = error?.message?.includes("RESOURCE_EXHAUSTED") || error?.status === "RESOURCE_EXHAUSTED" || error?.code === 429;
      if (isQuotaError && i < maxRetries - 1) {
        console.warn(`Quota exceeded, retrying in ${delay}ms... (Attempt ${i + 1}/${maxRetries})`);
        await new Promise(resolve => setTimeout(resolve, delay));
        delay *= 2; // Exponential backoff
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
    resolution?: string;
    format?: string;
    artifacts?: string[];
  };
  heatmaps?: Record<string, string>;
}

export type HeatmapMode = 'artifacts' | 'noise' | 'lighting';

export async function generateHeatmap(base64Image: string, mode: HeatmapMode = 'artifacts'): Promise<string> {
  const model = "gemini-2.5-flash-image";
  
  const prompts = {
    artifacts: "Analyze this image for deepfake artifacts. Generate a heatmap overlay highlighting the areas that look AI-generated or manipulated. Use a glowing red-to-yellow thermal gradient for the manipulated regions and keep the original image context visible in the background. Return only the resulting heatmap image.",
    noise: "Perform a noise level analysis on this image. Generate a high-contrast heatmap where inconsistent pixel noise patterns (common in AI generation) are highlighted in bright cyan and purple. This should look like an Error Level Analysis (ELA) map. Return only the resulting image.",
    lighting: "Analyze the lighting and shadows in this image. Generate a heatmap that highlights inconsistencies in light direction and shadow casting, which often occur in deepfakes. Use a high-contrast blue-to-white gradient to show light vectors. Return only the resulting image."
  };

  const response: GenerateContentResponse = await withRetry(() => ai.models.generateContent({
    model,
    contents: {
      parts: [
        { inlineData: { data: base64Image.split(',')[1], mimeType: "image/jpeg" } },
        { text: prompts[mode] }
      ]
    }
  }));

  for (const part of response.candidates?.[0]?.content?.parts || []) {
    if (part.inlineData) {
      return `data:image/png;base64,${part.inlineData.data}`;
    }
  }
  
  throw new Error(`${mode} heatmap generation failed.`);
}

export async function analyzeImage(base64Image: string): Promise<AnalysisResult> {
  const model = "gemini-3.1-pro-preview";
  
  const prompt = `
    Analyze this image for authenticity. 
    Determine if it is a real photograph or AI-generated (Deepfake/GenAI).
    Provide the following in JSON format:
    {
      "isReal": boolean,
      "confidence": number (0-1),
      "source": "string (e.g., 'Phone Camera', 'DSLR', 'Stable Diffusion', 'Midjourney', 'DALL-E')",
      "reasoning": "string (detailed explanation of artifacts, lighting, or metadata clues)",
      "metadata": {
        "artifacts": ["string array of detected anomalies"]
      }
    }
  `;

  const response: GenerateContentResponse = await withRetry(() => ai.models.generateContent({
    model,
    contents: {
      parts: [
        { text: prompt },
        { inlineData: { data: base64Image.split(',')[1], mimeType: "image/jpeg" } }
      ]
    },
    config: {
      responseMimeType: "application/json"
    }
  }));

  try {
    return JSON.parse(response.text || "{}");
  } catch (e) {
    console.error("Failed to parse Gemini response", e);
    throw new Error("Analysis failed to return valid data.");
  }
}
