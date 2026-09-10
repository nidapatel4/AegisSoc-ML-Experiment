// Proxy to the Python ML service's batch anomaly analysis.
// POST { events: [{...}, ...] }

import { NextResponse } from "next/server";
import { analyzeEvents, mlServiceError } from "@/lib/ml-service";

export const runtime = "nodejs";

const MAX_EVENTS = 500;

export async function POST(request: Request) {
  let events: Record<string, unknown>[];
  try {
    const body = (await request.json()) as { events?: unknown };
    if (!Array.isArray(body.events)) {
      return NextResponse.json({ error: "Provide an { events } array." }, { status: 400 });
    }
    if (body.events.length === 0) {
      return NextResponse.json({ error: "The events array is empty." }, { status: 400 });
    }
    if (body.events.length > MAX_EVENTS) {
      return NextResponse.json(
        { error: `Batch is limited to ${MAX_EVENTS} events per call.` },
        { status: 413 },
      );
    }
    events = body.events.filter(
      (e): e is Record<string, unknown> => typeof e === "object" && e !== null,
    );
  } catch {
    return NextResponse.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  try {
    const results = await analyzeEvents(events);
    return NextResponse.json(results);
  } catch (err) {
    return NextResponse.json({ error: mlServiceError(err) }, { status: 502 });
  }
}
