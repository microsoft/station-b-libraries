from dataclasses_json.api import DataClassJsonMixin
from datetime import date
from datetime import datetime
from datetime import time
from typing import *

DT = TypeVar('DT', date, time, datetime)
K = TypeVar('K')
T = TypeVar('T')
V = TypeVar('V')
class DiscriminatorDecoderError(ValueError):
    pass

class UnregisteredDiscriminatorTypeError(DiscriminatorDecoderError):
    pass

def discriminator_decoder(discriminator_key: str, mappings: Dict[str, Type[T]], *, default_factory: Union[Callable[[], T], Type[T]] = None) -> Callable[[Dict[str, Any]], T]:
    lst_gen = lambda: ', '.join(f"'{t}'" for t in mappings.keys())
    def decoder(data: Dict[str, Any]) -> T:
        if (not isinstance(data, dict)):
            raise DiscriminatorDecoderError(f"A dict-like object is expected to decode any of [ {lst_gen()} ], got {type(data)}")
        elif (discriminator_key not in data):
            raise DiscriminatorDecoderError(f"Discriminator field '{discriminator_key}' was not presented in the body: '{data}'")
        elif (data[discriminator_key] not in mappings):
            raise UnregisteredDiscriminatorTypeError(f"Discriminator field '{discriminator_key}' has invalid value '{data[discriminator_key]}'")
        else:
            return mappings[data[discriminator_key]].from_dict(data)
    
    if (default_factory is not None):
        def safe_decoder(data: Dict[str, Any]) -> Optional[T]:
            try:
                return decoder(data)
            except DiscriminatorDecoderError:
                if (isinstance(default_factory, type) and issubclass(default_factory, DataClassJsonMixin)):
                    return default_factory.from_dict(data)
                else:
                    return default_factory()
        result = safe_decoder
    
    else:
        result = decoder
    
    return result

def datetime_decoder(cls: Type[DT]) -> Callable[[str], DT]:
    def decoder(s: str) -> DT:
        if (not isinstance(s, str)):
            raise ValueError(f"Unable to decode {cls.__name__}: expected str, got '{s}' ({type(s)})")
        return cls.fromisoformat(s.replace('Z', '+00:00'))
    return decoder


__all__ = \
[
    'DiscriminatorDecoderError',
    'UnregisteredDiscriminatorTypeError',
    'datetime_decoder',
    'discriminator_decoder',
]
