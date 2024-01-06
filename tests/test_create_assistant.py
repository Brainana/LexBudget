from scripts import create_assistant


def test_get_file_paths():
    file_paths = create_assistant.get_file_paths("sample_budget_docs", "pdf")
    assert len(file_paths) == 2
    assert all(file_path.endswith("pdf") for file_path in file_paths)
    # Check case-agnostic
    file_paths = create_assistant.get_file_paths("sample_budget_docs", "PDF")
    assert len(file_paths) == 2
    assert all(file_path.endswith("pdf") for file_path in file_paths)
