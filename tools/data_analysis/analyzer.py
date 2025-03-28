# ai_agent_framework/tools/data_analysis/analyzer.py

"""
Data Analysis Tool using Pandas

This module provides a tool for performing basic data analysis tasks
on structured data provided as text (e.g., CSV, JSON).
"""

import asyncio
import io
import json
import logging
from typing import Dict, Optional, List, Literal, Union, Any

# Attempt pandas import and handle if missing
try:
    import pandas as pd
except ImportError:
    pd = None # Set pandas to None if not installed

# Framework components
from ai_agent_framework.core.tools.base import BaseTool
from ai_agent_framework.core.exceptions import ToolError

logger = logging.getLogger(__name__)

# Define supported analysis types
AnalysisType = Literal[
    "summary_stats", # df.describe()
    "correlation",   # df.corr()
    "column_list",   # df.columns
    "data_types",    # df.dtypes
    "value_counts",  # df[column].value_counts()
    "head",          # df.head()
    "shape",         # df.shape
    "missing_values" # df.isnull().sum()
]

DataFormat = Literal["csv", "json", "dict"] # Supported input data formats

class DataAnalysisTool(BaseTool):
    """
    Tool for analyzing structured data using the pandas library.

    Accepts data as a string (CSV or JSON) or a list of dictionaries,
    performs specified analysis, and returns the results.
    Requires the 'pandas' library to be installed.
    """

    def __init__(
        self,
        name: str = "data_analyzer",
        description: str = ("Performs basic data analysis (like summary statistics, "
                          "correlation, column info) on provided structured data (CSV/JSON string or dict)."),
        max_rows: int = 10000, # Limit rows to prevent excessive memory usage
        max_cols: int = 100,  # Limit columns
        **kwargs # Catch-all for potential future BaseTool args
    ):
        """
        Initialize the Data Analysis tool.

        Args:
            name: Name of the tool.
            description: Description of what the tool does.
            max_rows: Maximum number of rows to process.
            max_cols: Maximum number of columns to process.
            **kwargs: Additional arguments passed to BaseTool.
        """
        if pd is None:
            raise ImportError("Pandas library is required for DataAnalysisTool. Please install it: pip install pandas")

        self.max_rows = max_rows
        self.max_cols = max_cols

        # Define parameters schema
        parameters_schema = {
            "type": "object",
            "properties": {
                "data_content": {
                    "type": ["string", "object"], # Accepts string (CSV/JSON) or list/dict
                    "description": "The data to analyze, provided as a CSV string, a JSON string (representing list of records or dict), or directly as a list/dictionary."
                },
                 "data_format": {
                    "type": "string",
                    "description": "Format of the data_content if provided as a string ('csv' or 'json'). Required if data_content is a string.",
                    "enum": ["csv", "json", "dict"] # 'dict' implies already parsed
                },
                "analysis_type": {
                    "type": "string",
                    "description": "The type of analysis to perform.",
                    "enum": [
                        "summary_stats", "correlation", "column_list",
                        "data_types", "value_counts", "head", "shape", "missing_values"
                    ]
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: List of column names to focus the analysis on (required for 'value_counts')."
                },
                "head_rows": {
                     "type": "integer",
                     "description": "Number of rows to return for 'head' analysis (default 5).",
                     "default": 5
                }
            },
            "required": ["data_content", "analysis_type"]
        }

        # Define examples
        examples = [
            {
                "description": "Get summary statistics for CSV data.",
                "parameters": {
                    "data_content": "col1,col2\n1,10\n2,20\n3,30",
                    "data_format": "csv",
                    "analysis_type": "summary_stats"
                },
                "result_summary": "DataFrame describing statistics of numerical columns."
            },
             {
                "description": "Get value counts for a specific column in JSON data.",
                "parameters": {
                    "data_content": '[{"user":"A","val":1},{"user":"B","val":2},{"user":"A","val":3}]',
                    "data_format": "json",
                    "analysis_type": "value_counts",
                    "columns": ["user"]
                },
                "result_summary": "Counts of unique values in the 'user' column."
            },
            {
                 "description": "Show the first 3 rows of data passed as dict/list.",
                 "parameters": {
                    "data_content": [{"colA": 1, "colB": "x"}, {"colA": 2, "colB": "y"}],
                    "data_format": "dict", # Indicate it's already parsed
                    "analysis_type": "head",
                    "head_rows": 3
                 },
                 "result_summary": "First 3 rows of the dataset."
            }
        ]

        super().__init__(
            name=name,
            description=description,
            parameters=parameters_schema,
            examples=examples,
            required_permissions=["data_analysis"], # Define permission needed
            **kwargs
        )

    def _parse_data(self, data_content: Union[str, List[Dict], Dict], data_format: Optional[DataFormat]) -> pd.DataFrame:
        """Parses input data into a pandas DataFrame."""
        df = None
        if isinstance(data_content, str):
            if not data_format:
                 raise ToolError("`data_format` ('csv' or 'json') is required when `data_content` is a string.")
            try:
                data_io = io.StringIO(data_content)
                if data_format == 'csv':
                    df = pd.read_csv(data_io)
                elif data_format == 'json':
                    # Try parsing as records (list of dicts) first, then other orientations
                    try:
                         df = pd.read_json(data_io, orient='records', lines=False)
                    except ValueError:
                         data_io.seek(0) # Reset buffer
                         try:
                              df = pd.read_json(data_io, orient='records', lines=True) # Try JSON lines
                         except ValueError:
                              data_io.seek(0) # Reset buffer
                              df = pd.read_json(data_io) # Try default orientation
                else:
                    raise ToolError(f"Unsupported string data_format: '{data_format}'. Use 'csv' or 'json'.")
            except pd.errors.ParserError as e:
                 raise ToolError(f"Failed to parse {data_format} data string: {e}")
            except Exception as e:
                 raise ToolError(f"Error reading {data_format} string data: {e}")

        elif isinstance(data_content, list) and all(isinstance(item, dict) for item in data_content):
             if data_format and data_format != 'dict':
                  logger.warning(f"Input is a list of dicts, but format specified as '{data_format}'. Processing as list of dicts.")
             df = pd.DataFrame(data_content)
        elif isinstance(data_content, dict): # Could be {col:[vals]} or other orientations
             if data_format and data_format != 'dict':
                  logger.warning(f"Input is a dict, but format specified as '{data_format}'. Processing as dict.")
             # Best effort conversion from dict
             try:
                 df = pd.DataFrame(data_content)
             except ValueError as e:
                 raise ToolError(f"Could not create DataFrame from dictionary structure: {e}")
        else:
             raise ToolError("Invalid `data_content` type. Must be CSV/JSON string, list of dicts, or dict.")

        if df is None: # Should not happen if logic above is correct, but defensive check
             raise ToolError("Could not load data into DataFrame.")

        # Apply row/column limits
        if df.shape[0] > self.max_rows:
            logger.warning(f"Input data has {df.shape[0]} rows, exceeding max {self.max_rows}. Truncating.")
            df = df.head(self.max_rows)
        if df.shape[1] > self.max_cols:
             logger.warning(f"Input data has {df.shape[1]} columns, exceeding max {self.max_cols}. Selecting first {self.max_cols}.")
             df = df.iloc[:, :self.max_cols]

        return df

    def _perform_analysis(
        self,
        df: pd.DataFrame,
        analysis_type: AnalysisType,
        columns: Optional[List[str]] = None,
        head_rows: int = 5
        ) -> Any:
        """Performs the requested analysis on the DataFrame."""

        # Validate columns if provided
        if columns:
             missing_cols = [col for col in columns if col not in df.columns]
             if missing_cols:
                  raise ToolError(f"Specified columns not found in data: {', '.join(missing_cols)}")
             df_subset = df[columns] # Use subset for analysis if columns specified
        else:
             df_subset = df

        logger.info(f"Performing analysis type: '{analysis_type}'")

        if analysis_type == "summary_stats":
            # Select only numeric columns for describe()
            numeric_df = df_subset.select_dtypes(include=[np.number])
            if numeric_df.empty:
                 return "No numeric columns found for summary statistics."
            return numeric_df.describe()
        elif analysis_type == "correlation":
            # Select only numeric columns for correlation
            numeric_df = df_subset.select_dtypes(include=[np.number])
            if numeric_df.shape[1] < 2:
                 return "At least two numeric columns are required for correlation analysis."
            return numeric_df.corr()
        elif analysis_type == "column_list":
            return df.columns.tolist()
        elif analysis_type == "data_types":
            return df.dtypes.apply(lambda x: str(x)).to_dict() # Convert dtypes to string for JSON
        elif analysis_type == "value_counts":
            if not columns:
                raise ToolError("'columns' parameter is required for 'value_counts' analysis.")
            if len(columns) != 1:
                 raise ToolError("Specify exactly one column for 'value_counts'.")
            col_name = columns[0]
            # Limit number of unique values shown?
            return df[col_name].value_counts() # .head(20) # Example limit
        elif analysis_type == "head":
            return df.head(head_rows)
        elif analysis_type == "shape":
            return {"rows": df.shape[0], "columns": df.shape[1]}
        elif analysis_type == "missing_values":
            return df.isnull().sum() # Returns series
        else:
            # This case should ideally be prevented by the parameter schema validation
            raise ToolError(f"Unsupported analysis type: {analysis_type}")

    def _format_result(self, analysis_result: Any) -> Union[str, Dict, List]:
         """Formats the analysis result for output."""
         if isinstance(analysis_result, pd.DataFrame):
             # Convert DataFrame to JSON string (records orientation)
             # Could also use .to_string(), .to_markdown(), etc.
             try:
                 return analysis_result.to_dict(orient='records')
             except Exception: # Fallback for complex types
                  return analysis_result.to_string()
         elif isinstance(analysis_result, pd.Series):
             # Convert Series to JSON string (dictionary)
             try:
                 return analysis_result.to_dict()
             except Exception: # Fallback for complex types
                  return analysis_result.to_string()
         elif isinstance(analysis_result, (dict, list)):
              # Already JSON serializable
              return analysis_result
         else:
              # Convert other types to string
              return str(analysis_result)


    async def _run(
        self,
        data_content: Union[str, List[Dict], Dict],
        analysis_type: AnalysisType,
        data_format: Optional[DataFormat] = None,
        columns: Optional[List[str]] = None,
        head_rows: int = 5
    ) -> Dict[str, Any]:
        """
        Performs data analysis asynchronously by running pandas operations in a thread.

        Args:
            data_content: The data (string or list/dict).
            analysis_type: Type of analysis requested.
            data_format: Format hint if data_content is string ('csv', 'json').
            columns: Optional list of columns for analysis.
            head_rows: Number of rows for 'head' analysis.

        Returns:
            Dictionary containing the analysis results or an error.
        """
        if pd is None:
             # This check should ideally happen in __init__, but double-check
             raise ToolError("Pandas library is not installed, cannot perform analysis.")

        try:
            # --- Run synchronous pandas code in a thread pool ---
            def sync_analysis():
                # 1. Parse data inside the sync function
                df = self._parse_data(data_content, data_format)
                # 2. Perform analysis
                analysis_raw = self._perform_analysis(df, analysis_type, columns, head_rows)
                # 3. Format result
                analysis_formatted = self._format_result(analysis_raw)
                return analysis_formatted

            # Use asyncio.to_thread for modern Python (3.9+)
            # Or loop.run_in_executor(None, sync_analysis) for older versions
            analysis_result = await asyncio.to_thread(sync_analysis)
            # ----------------------------------------------------

            return {
                "status": "success",
                "analysis_type": analysis_type,
                "result": analysis_result
            }
        except (ToolError, ValueError, TypeError) as e:
             # Catch specific errors raised by our helper methods
             logger.error(f"Data analysis failed for type '{analysis_type}': {e}", exc_info=True)
             raise ToolError(f"Data analysis failed: {e}") from e # Re-raise as ToolError
        except Exception as e:
             # Catch unexpected errors during analysis
             logger.exception(f"Unexpected error during data analysis type '{analysis_type}': {e}")
             raise ToolError(f"An unexpected error occurred during data analysis: {e}")