; Standard function definitions
(function_definition
  name: (identifier) @function.name
  body: (block) @function.body) @function.def

; Class definitions
(class_definition
  name: (identifier) @class.name
  body: (block) @class.body) @class.def

; Imports
(import_statement) @import
(import_from_statement) @import
