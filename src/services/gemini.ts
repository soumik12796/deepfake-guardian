import { GoogleGenAI, GenerateContentResponse } from "@google/genai";

export type HeatmapMode = 'artifacts' | 'noise' | 'lighting';

const API_KEY = import.meta.env.VITE_API_KEY;

const ai = new GoogleGenAI({ apiKey: API_KEY });

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
  const model ="gemini-2.5-flash";

  const prompt = `Analyze this image and return ONLY valid JSON, no extra text:
{
  "isReal": true or false,
  "confidence": number between 0 and 1,
  "source": "AI Generated" or "Real Photo",
  "reasoning": "brief explanation",
  "metadata": {
    "artifacts": []
  }
}`;

  const response: GenerateContentResponse = await ai.models.generateContent({
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
  });

  const rawText = response.text || "";
  const cleanText = rawText.replace(/```json/g, "").replace(/```/g, "").trim();

  try {
    return JSON.parse(cleanText);
  } catch (e) {
    throw new Error("Analysis failed");
  }
}

export async function generateHeatmap(
  base64Image: string,
  mode: HeatmapMode
): Promise<string> {
  const model ="gemini-2.5-flash";

  const modePrompts: Record<HeatmapMode, string> = {
    artifacts: "Describe visual artifacts in this image.",
    noise: "Describe noise patterns in this image.",
    lighting: "Describe lighting inconsistencies in this image.",
  };

  const response: GenerateContentResponse = await ai.models.generateContent({
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
  });

  return response.text || "No analysis available";
}