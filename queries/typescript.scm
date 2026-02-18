; Standard functions
(function_declaration
  name: (identifier) @function.name
  body: (statement_block) @function.body) @function.def

; Classes
(class_declaration
  name: (type_identifier) @class.name
  body: (class_body) @class.body) @class.def

; Interfaces (Treat as classes for the graph)
(interface_declaration
  name: (type_identifier) @class.name
  body: (object_type) @class.body) @class.def

; Methods
(method_definition
  name: (property_identifier) @function.name
  body: (statement_block) @function.body) @function.def

; Arrow functions
(lexical_declaration
  (variable_declarator
    name: (identifier) @function.name
    value: (arrow_function
      body: (statement_block) @function.body))) @function.def

; Imports
(import_statement) @import
