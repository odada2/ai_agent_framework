# ai_agent_framework/tools/data_analysis/__init__.py

"""
Data Analysis Tools Package

Provides tools for analyzing structured data, primarily using pandas.
"""

# Import pandas conditionally to allow the package to load even if pandas isn't installed,
# but the tool itself will raise an error in __init__ if it is used without pandas.
try:
    import pandas as pd
    PANDAS_INSTALLED = True
except ImportError:
    PANDAS_INSTALLED = False
    pd = None # Define pd as None so the import below doesn't fail immediately

# Only import the tool if pandas is available, otherwise attempting to use it
# will raise an error later anyway.
if PANDAS_INSTALLED:
    from .analyzer import DataAnalysisTool
    __all__ = ['DataAnalysisTool']
else:
     # If pandas is not installed, __all__ is empty, so the tool won't be easily accessible.
     # Trying to import it directly will fail later if pandas is missing.
    __all__ = []
    print("WARNING: pandas library not found. DataAnalysisTool will not be available.")