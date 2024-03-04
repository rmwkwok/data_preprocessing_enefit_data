'''Data class customed to the Enefit dataset.'''

from typing import Optional


from dataclasses import dataclass


from pandas import DataFrame
# from utils import stat


# @dataclass
# class FocusableBase:
#     '''Make a class focusable, meaning we can pre-designate an object
#     to which any method call to the class object will be re-directed.
#     This, for example, enables us to do one chain of methods to the class
#     for modifying multiple dataframe objects. This is for fun.
#     '''
#     def __post_init__(self, focusable_attributes) -> None:
#         self._focus = None
#         self._buffer = None
#         self._focusable_attrs = focusable_attributes

#     def __getattr__(self, attr) -> 'FocusableBase' | Callable:
#         if self._focus is not None:
#             ret = getattr(self._buffer, attr)
#             if callable(ret):
#                 def _wrapper(*args, **kwargs):
#                     self._buffer = ret(*args, **kwargs)
#                     return self
#                 return _wrapper
#             else:
#                 self._buffer = ret
#                 return self
#         else:
#             raise AttributeError(f'No attribute "{attr}".')

#     def set_focus(self, attr: Optional[str]=None) -> 'FocusableBase':
#         if attr is None:
#             self._focus = None
#             self._buffer = None
#         elif attr in self._focusable_attrs:
#             self._focus = attr
#             self._buffer = getattr(self, attr)
#         else:
#             raise ValueError(
#                 f'Must be None or one of {self._focusable_attrs}. Got {attr}.'
#             )
#         return self

#     def save_focus(self, attr: Optional[str]=None) -> 'FocusableBase':
#         if attr is None:
#             attr = self._focus
#         elif (
#             (not isinstance(attr, str)) or
#             (hasattr(self, attr) and (attr not in self._focusable_attrs))
#         ):
#             raise ValueError(
#                 f'attr must be either None, one of {self._focusable_attrs}, '
#                 f'or an unused attribute name string. Got "{attr}".'
#             )
#         setattr(self, attr, self._buffer)
#         return self


@dataclass
class Data:
# class Data(FocusableBase):
    targets: Optional[DataFrame] = None
    rev_tar: Optional[DataFrame] = None
    clients: Optional[DataFrame] = None
    his_wea: Optional[DataFrame] = None
    for_wea: Optional[DataFrame] = None
    ele_prc: Optional[DataFrame] = None
    gas_prc: Optional[DataFrame] = None
    sam_pre: Optional[DataFrame] = None

    # def __post_init__(self) -> None:
    #     self._dfs = [
    #         'targets', 'rev_tar', 'clients', 'his_wea',
    #         'for_wea', 'ele_prc', 'gas_prc', 'sam_pre',
    #     ]
    #     # super().__post_init__(focusable_attributes=self._dfs.copy())


    # def __str__(self) -> str:
    #     return '\n'.join(
    #         f'{name}: {stat(getattr(self, name))}' for name in self._dfs
    #     )

    # def __repr__(self) -> str:
    #     return self.__str__()