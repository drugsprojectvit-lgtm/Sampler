(function_declaration
  name: (identifier) @function.name) @function.def

(type_declaration
  (type_spec
    name: (type_identifier) @class.name)) @class.def

(import_declaration) @import
