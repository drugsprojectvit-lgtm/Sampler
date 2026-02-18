; Functions: func Foo() {}
(function_declaration
  name: (identifier) @function.name
  body: (block) @function.body) @function.def

; Methods: func (s *Struct) Foo() {}
(method_declaration
  name: (field_identifier) @function.name
  body: (block) @function.body) @function.def

; Structs and Interfaces (The "Classes" of Go)
(type_declaration
  (type_spec
    name: (type_identifier) @class.name
    type: (struct_type))) @class.def

(type_declaration
  (type_spec
    name: (type_identifier) @class.name
    type: (interface_type))) @class.def

; Imports
(import_declaration) @import
