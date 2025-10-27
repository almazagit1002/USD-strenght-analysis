from typing import Dict, Type
import logging
from .fetchers import BaseFetcher, StooqFetcher, FredFetcher

class FetcherFactory:
    """Factory to create appropriate fetcher based on source type"""
    
    _fetchers: Dict[str, Type[BaseFetcher]] = {
        'stooq': StooqFetcher,
        'fred': FredFetcher
    }
    
    @classmethod
    def create_fetcher(cls, source_type: str, base_url: str, logger: logging.Logger) -> BaseFetcher:
        """Create a fetcher instance based on source type"""
        if source_type not in cls._fetchers:
            raise ValueError(f"Unknown source type: {source_type}")
        
        fetcher_class = cls._fetchers[source_type]
        
        if source_type == 'stooq':
            return fetcher_class(base_url, logger)
        elif source_type == 'fred':
            return fetcher_class(logger)  # FRED doesn't need base_url
        
    @classmethod
    def register_fetcher(cls, source_type: str, fetcher_class: Type[BaseFetcher]):
        """Register a new fetcher type"""
        cls._fetchers[source_type] = fetcher_class