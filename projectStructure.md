# SMC Trading Bot - Professional Project Structure

smc_trading_bot/
├── README.md
├── requirements.txt
├── setup.py
├── .env.example
├── .gitignore
├── docker-compose.yml
├── pytest.ini
├── config/
│   ├── __init__.py
│   ├── settings.py
│   ├── mt5_config.py
│   └── logging_config.py
├── data_service/
│   ├── __init__.py
│   ├── data_fetcher.py          # Your existing MT5 fetcher (enhanced)
│   ├── data_validator.py
│   ├── data_preprocessor.py
│   └── tests/
│       ├── __init__.py
│       ├── test_data_fetcher.py
│       └── test_data_validator.py
├── market_structure/
│   ├── __init__.py
│   ├── structure_detector.py
│   ├── swing_analyzer.py
│   ├── bos_choc_detector.py
│   ├── trend_classifier.py
│   └── tests/
│       ├── __init__.py
│       ├── test_structure_detector.py
│       ├── test_swing_analyzer.py
│       └── test_bos_choc_detector.py
├── pattern_recognition/
│   ├── __init__.py
│   ├── fair_value_gap.py
│   ├── order_block_detector.py
│   ├── inducement_finder.py
│   ├── institutional_candles.py
│   └── tests/
│       ├── __init__.py
│       ├── test_fvg.py
│       └── test_order_blocks.py
├── liquidity_service/
│   ├── __init__.py
│   ├── liquidity_detector.py
│   ├── sweep_analyzer.py
│   ├── equal_levels.py
│   └── tests/
│       ├── __init__.py
│       └── test_liquidity.py
├── premium_discount/
│   ├── __init__.py
│   ├── pd_calculator.py
│   ├── fibonacci_levels.py
│   ├── array_hierarchy.py
│   └── tests/
│       ├── __init__.py
│       └── test_pd_calculator.py
├── timeframe_coordinator/
│   ├── __init__.py
│   ├── multi_tf_sync.py
│   ├── conflict_resolver.py
│   └── tests/
│       ├── __init__.py
│       └── test_coordinator.py
├── risk_management/
│   ├── __init__.py
│   ├── position_sizer.py
│   ├── stop_loss_manager.py
│   ├── risk_calculator.py
│   └── tests/
│       ├── __init__.py
│       └── test_risk_manager.py
├── backtesting_engine/
│   ├── __init__.py
│   ├── backtester.py
│   ├── performance_analyzer.py
│   ├── trade_simulator.py
│   └── tests/
│       ├── __init__.py
│       └── test_backtester.py
├── analysis_dashboard/
│   ├── __init__.py
│   ├── visualizer.py
│   ├── chart_plotter.py
│   ├── performance_dashboard.py
│   └── static/
│       ├── css/
│       ├── js/
│       └── templates/
├── trading_executor/
│   ├── __init__.py
│   ├── order_manager.py
│   ├── execution_engine.py
│   └── tests/
│       ├── __init__.py
│       └── test_executor.py
├── monitoring/
│   ├── __init__.py
│   ├── system_monitor.py
│   ├── performance_tracker.py
│   ├── alerts.py
│   └── logs/
├── ml_models/
│   ├── __init__.py
│   ├── pattern_classifier.py
│   ├── context_analyzer.py
│   ├── model_trainer.py
│   ├── models/
│   └── tests/
│       ├── __init__.py
│       └── test_models.py
├── utils/
│   ├── __init__.py
│   ├── helpers.py
│   ├── constants.py
│   ├── decorators.py
│   └── exceptions.py
├── scripts/
│   ├── setup_project.py
│   ├── run_tests.py
│   ├── deploy.py
│   └── data_collection.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_integration.py
│   └── fixtures/
│       ├── sample_data.csv
│       └── test_scenarios.json
└── docs/
    ├── architecture.md
    ├── api_reference.md
    ├── user_guide.md
    └── deployment.md