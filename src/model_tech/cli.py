from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import typer
from rich import print

from model_tech.config import DataConfig, LabelingConfig, TrainConfig, project_paths
from model_tech.data.store import ensure_dirs
from model_tech.data.symbols import normalize_symbol
from model_tech.data.update import update_symbol_ohlcv
from model_tech.infer import predict_signal
from model_tech.train import train_pipeline


app = typer.Typer(add_completion=False, help="Binance 4h TA signal model (BUY/SELL/HOLD).")


@app.command()
def download(
    symbols: str = typer.Option(
        "",
        help="Comma-separated symbols (default: config symbols). Example: BTCUSDT,ETHUSDT",
    ),
    since: str = typer.Option(
        "2021-01-01",
        help="Start date (YYYY-MM-DD) in UTC for initial history backfill.",
    ),
    data_dir: Optional[Path] = typer.Option(None, help="Override data directory."),
):
    paths = project_paths()
    if data_dir is not None:
        paths = type(paths)(root=paths.root, data_dir_override=data_dir)
    ensure_dirs(paths)

    cfg = DataConfig()
    sym_list = [normalize_symbol(s.strip(), default_quote="USDT") for s in symbols.split(",") if s.strip()] or list(cfg.symbols_default)

    start = date.fromisoformat(since)
    for sym in sym_list:
        print(f"[bold]Downloading[/bold] {sym} 4h since {start.isoformat()} ...")
        update_symbol_ohlcv(symbol=sym, start_date=start, data_cfg=cfg, paths=paths)

    print("[green]Done.[/green]")


@app.command()
def train(
    symbols: str = typer.Option("", help="Comma-separated symbols; default from config."),
    n_folds: int = typer.Option(6, help="Walk-forward folds."),
    mode: str = typer.Option("quality", help="Training mode: quality|fast"),
):
    paths = project_paths()
    ensure_dirs(paths)
    data_cfg = DataConfig()
    lab_cfg = LabelingConfig()
    tr_cfg = TrainConfig(n_folds=n_folds, mode=mode)

    sym_list = [normalize_symbol(s.strip(), default_quote="USDT") for s in symbols.split(",") if s.strip()] or list(data_cfg.symbols_default)
    out = train_pipeline(symbols=sym_list, paths=paths, data_cfg=data_cfg, lab_cfg=lab_cfg, tr_cfg=tr_cfg)
    print(out)


@app.command()
def predict(
    symbol: str = typer.Option(..., help="Symbol like BTCUSDT"),
    refresh: bool = typer.Option(False, help="If set, fetch/update candles on demand before predicting."),
):
    paths = project_paths()
    ensure_dirs(paths)
    sym = normalize_symbol(symbol, default_quote="USDT")
    result = predict_signal(symbol=sym, paths=paths, ensure_data=bool(refresh))
    print(result)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    workers: int = typer.Option(1, help="Uvicorn workers (use 1 for in-memory job queue)"),
    reload: bool = typer.Option(False, help="Auto-reload code (dev only)"),
):
    """
    Run HTTP API (FastAPI + Uvicorn).
    """
    import uvicorn

    uvicorn.run(
        "model_tech.api.app:app",
        host=host,
        port=int(port),
        workers=int(workers),
        reload=bool(reload),
    )


if __name__ == "__main__":
    app()


