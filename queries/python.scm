(function_definition
  name: (identifier) @function.name
  body: (block) @function.body) @function.def

(class_definition
  name: (identifier) @class.name) @class.def

(import_statement) @import
(import_from_statement) @import
