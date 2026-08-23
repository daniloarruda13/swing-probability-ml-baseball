import json
from pathlib import Path
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_notebook_is_valid_and_identified_as_historical(self):
        notebook_path = ROOT / "swing_probability_modeling.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreater(sum(cell["cell_type"] == "code" for cell in notebook["cells"]), 0)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("historical", readme.casefold())

    def test_project_metadata_and_requirements_agree(self):
        metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(metadata["project"]["name"], "swing-probability-baseball")
        self.assertIn("analysis", metadata["project"]["optional-dependencies"])
        self.assertEqual((ROOT / "requirements.txt").read_text().strip(), "-e .[analysis]")

    def test_readme_uses_real_repository_and_notebook_names(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("https://https", readme)
        self.assertIn("swing_probability_modeling.ipynb", readme)
        self.assertNotIn("swing_probability.ipynb", readme)


if __name__ == "__main__":
    unittest.main()
