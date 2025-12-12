# S03 – eda_cli: мини-EDA для CSV

Небольшое CLI-приложение для базового анализа CSV-файлов.
Используется в рамках Семинара 03 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (S03):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## Запуск CLI

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`);
- `--out-dir ` - папка для сохранения отчета (по умолчанию `reports`);
- `--max-hist-columns` - максимальное количество гистограмм для числовых колонок (по умолчанию `6`);
- `--top-k-categories` - количество топовых значений для категориальных признаков (по умолчанию `5`);
- `--title` - заголовок отчета (по умолчанию `Отчет`);
- `--min-missing-share` - порог доли пропусков (0.0-1.0) для выделения проблемных колонок (по умолчанию `0.5`).

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports
```

В результате в каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.

```bash
uv run eda-cli report data/customers.csv \
    --title "Анализ клиентской базы" \
    --out-dir "customer_analysis" \
    --max-hist-columns 10 \
    --top-k-categories 8 \
    --min-missing-share 0.2
```

В результате в каталоге `customer_analysis/` появятся:

- `report.md` – основной отчёт в Markdown с заголовком "Анализ клиентской базы";
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `problematic_columns.csv` – колонки с пропусками ≥ 20%;
- `top_categories/*.csv` – top-8 категорий по строковым признакам;
- `hist_*.png` – гистограммы для до 10 числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.

## Тесты

```bash
uv run pytest -q
```

## HTTP API

Проект также предоставляет REST API на базе FastAPI.

### Запуск сервера

```bash
uv run uvicorn eda_cli.api:app --reload --port 8000
```

## Эндпоинты

### Проверка работоспособности сервиса.

GET /health

```bash
curl http://localhost:8000/health
```

Оценка качества датасета по CSV-файлу.

POST /quality-from-csv

```bash
curl -F "file=@data/example.csv" http://localhost:8000/quality-from-csv
```

Возвращает первые N строк датасета.

POST /quality-flags-from-csv

```bash
curl -F "file=@data/example.csv" http://localhost:8000/quality-flags-from-csv
```

Возвращает полный набор флагов качества.