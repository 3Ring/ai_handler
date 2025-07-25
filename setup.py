from setuptools import setup, find_packages

setup(
    name="ai_handler",
    version="0.1.3",
    packages=find_packages(),
    install_requires=[],
    python_requires=">=3.13",
    extras_require={
        "openai": ["openai>=1.0"],
        "gemini": ["google-genai>=1.25.0"],
        "anthropic": ["anthropic>=0.6"],
    },    
)
