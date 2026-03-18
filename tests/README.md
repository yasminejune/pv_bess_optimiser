# Tests

This directory contains unit and integration tests for the ORS system.

## Running Tests

### Run all tests
```bash
pytest
```

### Run with verbose output
```bash
pytest -v
```

### Run specific test file
```bash
pytest tests/unit/services/test_pv_status.py
```

### Run specific test class or function
```bash
pytest tests/unit/services/test_pv_status.py::TestUpdatePVState
pytest tests/unit/services/test_pv_status.py::TestUpdatePVState::test_normal_generation
```

### Run with coverage report
```bash
pytest --cov=src --cov-report=html
```

## Test Structure

```
tests/
├── unit/               # Unit tests (isolated, fast)
│   └── services/
│       └── test_pv_status.py
└── integration/        # Integration tests (cross-module)
```

## PV Status Tests Coverage

The PV status service tests (`test_pv_status.py`) provide comprehensive coverage of:

### Core Functionality
- Normal generation with correct kWh conversion for 15-minute timestep
- Energy calculation for different timestep durations (15, 30, 60 minutes)
- Zero generation (valid night-time scenario)

### Data Quality Handling
- Missing generation telemetry → zero output with quality flag
- Negative generation → clamped to zero with flag
- Generation above rated power → clamped with flag
- Generation below minimum threshold → raised with flag

### Radiance-Based Estimation
- Energy estimation formula correctness (`estimate_energy_from_radiance`)
- Fallback to radiance when telemetry missing
- Derived generation_kw from estimated energy
- Quality flags: `missing_generation` and `estimated_from_radiance`
- Preference for telemetry over radiance when both available
- Validation of all radiance estimation parameters

### Export and Curtailment
- Exportable power respects `max_export_kw` limit
- Curtailment logic when generation exceeds export limit
- Curtailment disabled behavior (cap applied but no curtailment)
- No export limit (unlimited export) scenario
- Generation exactly at max_export (no curtailment triggered)

### Constraint Enforcement
- Rated power limit enforcement
- Minimum generation constraint
- Multiple constraints simultaneously
- Curtailment combined with power limits

### Validation
- Timestep validation (must be positive)
- Panel efficiency validation (0–1 range)
- Solar radiance validation (non-negative)
- Panel area validation (positive)

### Test Fixtures
- `base_spec`: Standard PV specification for testing
- `spec_with_radiance_params`: Spec with panel parameters for estimation
- `base_timestamp`: Fixed timestamp (2026-01-01 00:00:00) for reproducibility

## Writing New Tests

When adding tests:
1. Use descriptive test names: `test_<scenario>_<expected_behavior>`
2. Follow the Arrange-Act-Assert pattern
3. Use pytest fixtures for common setup
4. Test both happy path and edge cases
5. Use `pytest.approx()` for floating-point comparisons
6. Include docstrings explaining what is being tested

Example:
```python
def test_new_scenario(self, base_spec: PVSpec, base_timestamp: datetime):
    """Test that <specific behavior> works correctly."""
    # Arrange
    telemetry = PVTelemetry(timestamp=base_timestamp, generation_kw=50.0)
    
    # Act
    state = update_pv_state(base_spec, telemetry, timestep_minutes=15)
    
    # Assert
    assert state.generation_kw == 50.0
    assert len(state.quality_flags) == 0
```

## Test Isolation

All PV tests are isolated and do not require:
- External APIs or network access
- Database connections
- File system I/O
- Real-time clocks (uses fixed timestamps)

This ensures fast, deterministic test execution.
