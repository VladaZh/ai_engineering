from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    compute_quality_flags,
    correlation_matrix,
    flatten_summary_for_print,
    missing_table,
    summarize_dataset,
    top_categories,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [10, 20, 30, None],
            "height": [140, 150, 160, 170],
            "city": ["A", "B", "A", None],
        }
    )


def test_summarize_dataset_basic():
    df = _sample_df()
    summary = summarize_dataset(df)

    assert summary.n_rows == 4
    assert summary.n_cols == 3
    assert any(c.name == "age" for c in summary.columns)
    assert any(c.name == "city" for c in summary.columns)

    summary_df = flatten_summary_for_print(summary)
    assert "name" in summary_df.columns
    assert "missing_share" in summary_df.columns


def test_missing_table_and_quality_flags():
    df = _sample_df()
    missing_df = missing_table(df)

    assert "missing_count" in missing_df.columns
    assert missing_df.loc["age", "missing_count"] == 1

    summary = summarize_dataset(df)
    flags = compute_quality_flags(summary, missing_df, df)
    assert 0.0 <= flags["quality_score"] <= 1.0


def test_correlation_and_top_categories():
    df = _sample_df()
    corr = correlation_matrix(df)
    # корреляция между age и height существует
    assert "age" in corr.columns or corr.empty is False

    top_cats = top_categories(df, max_columns=5, top_k=2)
    assert "city" in top_cats
    city_table = top_cats["city"]
    assert "value" in city_table.columns
    assert len(city_table) <= 2

from eda_cli.core import (
    summarize_dataset,
    missing_table,
    compute_quality_flags,
    is_equal,
    is_constant,
    has_high_cardinality_categoricals,
)


def test_is_equal_with_duplicate_columns():
    """Тест проверяет обнаружение дублирующихся колонок"""
    # Создаем DataFrame с дублирующимися колонками
    df = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': ['a', 'b', 'c', 'd', 'e'],
        'col3': [1, 2, 3, 4, 5],  # Дубликат col1
        'col4': [10, 20, 30, 40, 50],
    })
    
    # Проверяем, что функция обнаруживает дубликаты
    assert is_equal(df) == True
    
    # DataFrame без дубликатов
    df_no_duplicates = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': ['a', 'b', 'c', 'd', 'e'],
        'col3': [10, 20, 30, 40, 50],
        'col4': [100, 200, 300, 400, 500],
    })
    
    assert is_equal(df_no_duplicates) == False


def test_is_constant_with_constant_column():
    """Тест проверяет обнаружение константных колонок"""
    # DataFrame с константной колонкой
    df_with_constant = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'status': ['active', 'active', 'active', 'active', 'active'],  # Константная
        'value': [10.5, 20.3, 15.7, 18.9, 12.4],
    })
    
    assert is_constant(df_with_constant) == True
    
    # DataFrame без константных колонок
    df_no_constant = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'status': ['active', 'inactive', 'pending', 'active', 'inactive'],
        'value': [10.5, 20.3, 15.7, 18.9, 12.4],
    })
    
    assert is_constant(df_no_constant) == False
    
    # DataFrame с константной колонкой, но с пропусками
    df_constant_with_nan = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'status': ['active', 'active', None, 'active', 'active'],
        'value': [10.5, 20.3, 15.7, 18.9, 12.4],
    })
    
    assert is_constant(df_constant_with_nan) == False


def test_has_high_cardinality_categoricals():
    """Тест проверяет обнаружение категориальных признаков с уникальными значениями для каждой строки"""
    # Создаем данные с уникальными значениями для каждой строки
    n_rows = 100
    df_high_cardinality = pd.DataFrame({
        'id': range(n_rows),
        'user_id': [f'user_{i}' for i in range(n_rows)],  # 100% уникальных значений
        'category': ['cat_' + str(i % 5) for i in range(n_rows)],  # 5% уникальных значений
        'value': [i * 10 for i in range(n_rows)],
    })
    
    # Должен обнаружить высокую кардинальность (100% уникальных значений)
    assert has_high_cardinality_categoricals(df_high_cardinality) == True
    
    # DataFrame с нормальной кардинальностью
    df_normal_cardinality = pd.DataFrame({
        'id': range(n_rows),
        'status': ['active' if i % 2 == 0 else 'inactive' for i in range(n_rows)],
        'category': ['cat_' + str(i % 10) for i in range(n_rows)],  # 10% уникальных значений
        'value': [i * 10 for i in range(n_rows)],
    })
    
    # 10% уникальных значений не превышает порог 70%
    assert has_high_cardinality_categoricals(df_normal_cardinality) == False


def test_compute_quality_flags_integration():
    """Интеграционный тест проверяет все эвристики вместе"""
    # Создаем DataFrame с разными проблемами
    df = pd.DataFrame({
        # Дубликат col1
        'col1': [1, 2, 3, 4, 5],
        'col2': [1, 2, 3, 4, 5],  # Дубликат col1
        
        # Константная колонка
        'constant_col': ['same', 'same', 'same', 'same', 'same'],
        
        # Категориальная с высокой кардинальностью
        'high_card_col': ['val1', 'val2', 'val3', 'val4', 'val5'],  # 100% уникальных
        
        # Нормальная колонка
        'normal_col': [10.5, 20.3, 15.7, 18.9, 12.4],
        
        # Колонка с пропусками
        'missing_col': [1.0, None, 3.0, None, 5.0],
    })
    
    # Вычисляем summary и missing для флагов
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    
    # Вычисляем флаги качества
    flags = compute_quality_flags(summary, missing_df, df)
    
    # Проверяем флаги
    assert flags['has_duplicate_columns'] == True
    assert flags['has_constant_columns'] == True
    assert flags['has_high_cardinality_categoricals'] == True
    
    # Проверяем, что оценка качества снижена из-за проблем
    assert flags['quality_score'] < 1.0
    
    # Проверяем расчет доли пропусков
    assert flags['max_missing_share'] == 0.4


def test_compute_quality_flags_with_good_data():
    """Тест проверяет флаги качества для хороших данных"""
    df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'name': ['Alice', 'Bob', 'Alice', 'Bob', 'Charlie', 'Charlie', 'David', 'David', 'Eve', 'Eve'],
        'age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        'salary': [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000],
        'department': ['IT', 'HR', 'IT', 'HR', 'IT', 'HR', 'Sales', 'Sales', 'Support', 'Support'],
    })
    
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags = compute_quality_flags(summary, missing_df, df)
    
    # Все флаги должны быть False для хороших данных
    assert flags['has_duplicate_columns'] == False
    assert flags['has_constant_columns'] == False
    assert flags['has_high_cardinality_categoricals'] == False
    
    # Оценка качества должна быть не ниже 0.5
    assert flags['quality_score'] >= 0.5

def test_compute_quality_flags_comprehensive():
    """
    Комплексный тест для проверки всех эвристик в compute_quality_flags.
    Проверяет флаги качества и связанные величины.
    """
    # 1. Создаем DataFrame для проверки константной колонки
    print("1. Тестирование константной колонки...")
    df_constant = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'constant_col': ['same', 'same', 'same', 'same', 'same'],  # Константная
        'normal_col': [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    
    summary_constant = summarize_dataset(df_constant)
    missing_constant = missing_table(df_constant)
    flags_constant = compute_quality_flags(summary_constant, missing_constant, df_constant)
    
    # Проверяем флаг константной колонки
    assert flags_constant['has_constant_columns'] == True, "Должен обнаружить константную колонку"
    assert flags_constant['has_duplicate_columns'] == False, f"Нет дублирующихся колонок. Флаги: {flags_constant}"
    assert flags_constant['has_high_cardinality_categoricals'] == False, "Нет высокой кардинальности"
    
    # Проверяем корректность связанных величин
    assert flags_constant['quality_score'] < 1.0, "Оценка должна снизиться из-за константной колонки"
    print(f"   Оценка качества с константной колонкой: {flags_constant['quality_score']:.2f}")
    
    # 2. Создаем DataFrame для проверки высокой кардинальности
    print("2. Тестирование высокой кардинальности...")
    df_high_card = pd.DataFrame({
        # Убираем числовые колонки, которые могут дублироваться
        'high_card_col': [f'val_{i}' for i in range(18)] + ['same', 'same'], 
        'category': ['A', 'B'] * 10,  # 2/20 = 10% < 70%
    })
    
    # Проверяем вручную
    assert df_high_card['high_card_col'].nunique() / len(df_high_card) > 0.7, "Должна быть высокая кардинальность"
    assert df_high_card['category'].nunique() / len(df_high_card) < 0.7, "Должна быть низкая кардинальность"
    
    summary_high_card = summarize_dataset(df_high_card)
    missing_high_card = missing_table(df_high_card)
    flags_high_card = compute_quality_flags(summary_high_card, missing_high_card, df_high_card)
    
    # Проверяем флаг высокой кардинальности
    assert flags_high_card['has_high_cardinality_categoricals'] == True, "Должен обнаружить высокую кардинальность"
    assert flags_high_card['has_constant_columns'] == False, "Нет константных колонок"
    assert flags_high_card['has_duplicate_columns'] == False, f"Нет дублирующихся колонок. Флаги: {flags_high_card}"
    
    # Проверяем, что оценка снижена
    assert flags_high_card['quality_score'] < 1.0, "Оценка должна снизиться из-за высокой кардинальности"
    print(f"   Оценка качества с высокой кардинальностью: {flags_high_card['quality_score']:.2f}")
    
    # 3. Создаем DataFrame для проверки дублирующихся колонок
    print("3. Тестирование дублирующихся колонок...")
    df_duplicate = pd.DataFrame({
        'id': [1, 2, 3],
        'col1': [10, 20, 30],
        'col2': [10, 20, 30],  # Дубликат col1
        'col3': ['a', 'b', 'c'],
        'col4': [100, 200, 300],
    })
    
    # Проверяем вручную
    assert df_duplicate['col1'].equals(df_duplicate['col2']), "col1 и col2 должны быть одинаковыми"
    
    summary_duplicate = summarize_dataset(df_duplicate)
    missing_duplicate = missing_table(df_duplicate)
    flags_duplicate = compute_quality_flags(summary_duplicate, missing_duplicate, df_duplicate)
    
    # Проверяем флаг дублирующихся колонок
    assert flags_duplicate['has_duplicate_columns'] == True, "Должен обнаружить дублирующиеся колонки"
    assert flags_duplicate['has_constant_columns'] == False, "Нет константных колонок"
    assert flags_duplicate['has_high_cardinality_categoricals'] == True, "Должен обнаружить высокую кардинальность"
    
    # Проверяем, что оценка снижена
    assert flags_duplicate['quality_score'] < 1.0, "Оценка должна снизиться из-за дублирующихся колонок"
    print(f"   Оценка качества с дублирующимися колонками: {flags_duplicate['quality_score']:.2f}")
    
    # 4. Создаем DataFrame с пропусками для проверки max_missing_share
    print("4. Тестирование пропусков...")
    df_missing = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'col_with_20_missing': [1.0, None, 3.0, None, 5.0],  # 40% пропусков
        'col_with_50_missing': [None, None, None, 4.0, 5.0],  # 60% пропусков
        'col_no_missing': [1, 2, 3, 4, 5],
    })
    
    # Проверяем вручную
    missing_counts = df_missing.isna().sum()
    assert missing_counts['col_with_20_missing'] == 2
    assert missing_counts['col_with_50_missing'] == 3
    
    summary_missing = summarize_dataset(df_missing)
    missing_missing = missing_table(df_missing)
    flags_missing = compute_quality_flags(summary_missing, missing_missing, df_missing)
    
    # Проверяем корректность расчета доли пропусков
    assert flags_missing['max_missing_share'] == 0.6, f"Макс. доля пропусков должна быть 0.6, а не {flags_missing['max_missing_share']}"
    assert flags_missing['too_many_missing'] == True, "Должен быть флаг too_many_missing при пропусках > 50%"
    
    # Проверяем корректность расчета в missing_table
    missing_counts_table = missing_missing['missing_count']
    missing_shares = missing_missing['missing_share']
    assert missing_counts_table['col_with_20_missing'] == 2, "col_with_20_missing должен иметь 2 пропуска"
    assert abs(missing_shares['col_with_20_missing'] - 0.4) < 0.01, f"col_with_20_missing должен иметь 40% пропусков, а имеет {missing_shares['col_with_20_missing']}"
    assert missing_counts_table['col_with_50_missing'] == 3, "col_with_50_missing должен иметь 3 пропуска"
    assert abs(missing_shares['col_with_50_missing'] - 0.6) < 0.01, f"col_with_50_missing должен иметь 60% пропусков, а имеет {missing_shares['col_with_50_missing']}"
    
    print(f"   Макс. доля пропусков: {flags_missing['max_missing_share']:.1%}")
    print(f"   Оценка качества с пропусками: {flags_missing['quality_score']:.2f}")
    
    # 5. Создаем DataFrame со всеми проблемами одновременно
    print("5. Тестирование всех проблем одновременно...")
    df_all_problems = pd.DataFrame({
        'col1': [1, 2, 3, 4, 5],
        'col2': [1, 2, 3, 4, 5],  # Дубликат col1
        'constant': ['A', 'A', 'A', 'A', 'A'],  # Константная
        'high_card': ['v1', 'v2', 'v3', 'v4', 'v5'],  # 100% уникальных
        'missing': [1, None, 3, None, None],  # 60% пропусков
        'normal': [10, 20, 30, 40, 50],
    })
    
    summary_all = summarize_dataset(df_all_problems)
    missing_all = missing_table(df_all_problems)
    flags_all = compute_quality_flags(summary_all, missing_all, df_all_problems)
    
    # Проверяем все флаги
    assert flags_all['has_duplicate_columns'] == True
    assert flags_all['has_constant_columns'] == True
    assert flags_all['has_high_cardinality_categoricals'] == True
    assert flags_all['max_missing_share'] == 0.6
    assert flags_all['too_many_missing'] == True
    
    # Проверяем, что оценка качества значительно снижена
    assert flags_all['quality_score'] < 0.5, f"Оценка должна быть низкой при всех проблемах, а равна {flags_all['quality_score']:.2f}"
    print(f"   Оценка качества со всеми проблемами: {flags_all['quality_score']:.2f}")
    
    # 6. Создаем идеальный DataFrame для проверки
    print("6. Тестирование идеальных данных...")
    df_perfect = pd.DataFrame({
        'id': range(1, 101), 
        'category': [f'cat_{i % 10}' for i in range(100)],  # 10% уникальных
        'value': [i * 10.0 for i in range(100)],
        'status': ['active' if i % 2 == 0 else 'inactive' for i in range(100)],  # 50% уникальных
    })
    
    summary_perfect = summarize_dataset(df_perfect)
    missing_perfect = missing_table(df_perfect)
    flags_perfect = compute_quality_flags(summary_perfect, missing_perfect, df_perfect)
    
    # Все флаги должны быть False
    assert flags_perfect['has_duplicate_columns'] == False, "Не должно быть дублирующихся колонок"
    assert flags_perfect['has_constant_columns'] == False, "Не должно быть константных колонок"
    assert flags_perfect['has_high_cardinality_categoricals'] == False, "Не должно быть высокой кардинальности"
    assert flags_perfect['too_few_rows'] == False, "Не должно быть флага too_few_rows"
    assert flags_perfect['too_many_missing'] == False, "Не должно быть флага too_many_missing"
    
    # Проверяем связанные величины
    assert flags_perfect['max_missing_share'] == 0.0, "Не должно быть пропусков"
    assert flags_perfect['quality_score'] > 0.9, f"Оценка должна быть высокой, а равна {flags_perfect['quality_score']:.2f}"
     
"""
    Запустить все тесты
    uv run pytest -q
"""