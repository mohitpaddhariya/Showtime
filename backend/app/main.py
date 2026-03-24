"""FastAPI application entry point."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.core.exceptions import ShowtimeError

app = FastAPI(
    title="Showtime",
    description="Turn rough screen recordings into polished demo videos.",
    version="0.1.0",
)

# CORS — allow common frontend dev servers + production origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

app.include_router(api_router, prefix="/api")


# ── Global error handlers ────────────────────────────────


@app.exception_handler(ShowtimeError)
async def showtime_error_handler(request: Request, exc: ShowtimeError):
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "detail": str(exc)},
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": "InternalServerError", "detail": str(exc)},
    )


# ── Health check ─────────────────────────────────────────


@app.get("/health")
def health_check():
    return {"status": "ok"}
