/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/runtime/contrib/cim/cim_runtime.cc
 * \brief A simple JSON runtime for CIM
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime::json;

class CIMRuntime : public JSONRuntimeBase {
 public:
  /*!
   * \brief The CIM runtime module. Deserialize the provided functions
   * on creation and store in the layer cache.
   *
   * \param symbol_name The name of the function.
   * \param graph_json serialized JSON representation of a sub-graph.
   * \param const_names The names of each constant in the sub-graph.
   */
  explicit CIMRuntime(const std::string& symbol_name, const std::string& graph_json,
                      const Array<String>& const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  /*!
   * \brief The type key of the module.
   *
   * \return module type key.
   */
  const char* type_key() const override { return "cim"; }

  /*!
   * \brief Initialize runtime. Create ACL layer from JSON
   * representation.
   *
   * \param consts The constant params from compiled model.
   */
  void Init(const Array<NDArray>& consts) override {
    CHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";
    SetupConstants(consts);
    BuildEngine();
  }

  void Run() override {
    LOG(FATAL) << "Cannot call run on CIM module without runtime enabled. "
               << "Please build with USE_CIM_GRAPH_RUNTIME.";
  }

  void BuildEngine() {
    LOG(WARNING) << "CIM engine is not initialized. "
                 << "Please build with USE_CIM_GRAPH_RUNTIME.";
  }

};

runtime::Module CIMRuntimeCreate(const String& symbol_name, const String& graph_json,
                                 const Array<String>& const_names) {
  auto n = make_object<CIMRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.CIMJSONRuntimeCreate").set_body_typed(CIMRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_cim")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<CIMRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

