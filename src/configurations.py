from typing import Dict, Optional, List, Union
from dataclasses import dataclass

@dataclass
class SourceConfig:
    """Configuration for a single data source"""
    type: str                             # "stooq", "fred", etc.
    base_url: str                        # URL or identifier for the source
    tickers: Dict[str, str]              # Tickers for this specific source

@dataclass
class DataSourceConfig:
    """Enhanced configuration class supporting multiple data sources"""
    fetch_name: str = "GenericData"
    
    # For backward compatibility - single source configs
    tickers: Optional[Dict[str, str]] = None
    base_url: Optional[str] = None
    
    # New multi-source support
    sources: Optional[List[SourceConfig]] = None
    
    # Optional DXY-like calculation
    dxy_base_constant: Optional[float] = None
    dxy_weights: Optional[Dict[str, float]] = None
    
    def is_multi_source(self) -> bool:
        """Check if this config uses multiple sources"""
        return self.sources is not None
    
    def get_all_tickers(self) -> Dict[str, str]:
        """Get all tickers from all sources combined"""
        if self.is_multi_source():
            all_tickers = {}
            for source in self.sources:
                all_tickers.update(source.tickers)
            return all_tickers
        else:
            return self.tickers or {}