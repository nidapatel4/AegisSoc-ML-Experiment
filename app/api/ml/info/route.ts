// Proxy to the Python ML service's model info (metadata + test metrics).

import { NextResponse } from "next/server";
import { getModelInfo, mlServiceError } from "@/lib/ml-service";

export const runtime = "nodejs";

export async function GET() {
  try {
    const info = await getModelInfo();
    return NextResponse.json(info);
  } catch (err) {
    return NextResponse.json({ error: mlServiceError(err) }, { status: 502 });
  }
}
