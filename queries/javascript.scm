; Standard functions: function foo() {}
(function_declaration
  name: (identifier) @function.name
  body: (statement_block) @function.body) @function.def

; Class definitions: class Foo {}
(class_declaration
  name: (identifier) @class.name
  body: (class_body) @class.body) @class.def

; Methods inside classes: class Foo { bar() {} }
(method_definition
  name: (property_identifier) @function.name
  body: (statement_block) @function.body) @function.def

; Arrow functions assigned to variables: const foo = () => {}
(lexical_declaration
  (variable_declarator
    name: (identifier) @function.name
    value: (arrow_function
      body: (statement_block) @function.body))) @function.def

; Imports
(import_statement) @import
