// ot_chat_apply: renders via libllama-common.so's common_chat_templates_apply,
// with the defaults llama-server's /apply-template applies for ollama.
// Returns prompt length, -(n+1) to resize, -1 render error, -2 invalid input
// (HTTP 400); error messages are written to buf.
#include "chat.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

extern "C" int ot_chat_apply(const void * model_v, const char * messages_json, const char * tools_json,
                             int think_set, int enable_thinking, const char * reasoning_effort,
                             char * buf, int buf_len) {
    const struct llama_model * model = (const struct llama_model *) model_v;
    try {
        common_chat_templates_inputs inputs;

        // https://github.com/ggml-org/llama.cpp/blob/b11351/tools/server/server-common.cpp#L1314-L1316
        try {
            inputs.messages = common_chat_msgs_parse_oaicompat(common_json::parse(messages_json));
            if (tools_json != nullptr && tools_json[0] != '\0') {
                inputs.tools = common_chat_tools_parse_oaicompat(common_json::parse(tools_json));
            }
        } catch (const std::exception & e) {
            if (buf_len > 0) { snprintf(buf, buf_len, "%s", e.what()); }
            return -2;
        }

        // ollama omits add_generation_prompt; the server default is true.
        // https://github.com/ggml-org/llama.cpp/blob/b11351/tools/server/server-common.cpp#L1322
        inputs.add_generation_prompt = true;

        // prefill_assistant default on: trailing assistant becomes a continuation.
        // https://github.com/ggml-org/llama.cpp/blob/b11351/tools/server/server-common.cpp#L1326-L1342
        if (!inputs.messages.empty() && inputs.messages.back().role == "assistant") {
            try {
                if (inputs.messages.size() >= 2 && inputs.messages[inputs.messages.size() - 2].role == "assistant") {
                    throw std::invalid_argument("Cannot have 2 or more assistant messages at the end of the list.");
                }
                if (!inputs.messages.back().tool_calls.empty()) {
                    throw std::invalid_argument("Cannot continue an assistant message that contains tool calls.");
                }
                inputs.continue_final_message = COMMON_CHAT_CONTINUATION_AUTO;
                inputs.add_generation_prompt  = false;
            } catch (const std::exception & e) {
                if (buf_len > 0) { snprintf(buf, buf_len, "%s", e.what()); }
                return -2;
            }
        }

        // llama-server defaults.
        // https://github.com/ggml-org/llama.cpp/blob/b11351/common/common.h#L651
        inputs.reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK;
        // https://github.com/ggml-org/llama.cpp/blob/b11351/common/arg.cpp#L961-L963
        inputs.chat_template_kwargs["preserve_reasoning"] = "true";

        // ollama's llamaServerChatTemplateKwargs: kwargs only when think is set.
        // Values are JSON-encoded (chat.cpp parses them with json::parse).
        // https://github.com/ollama/ollama/blob/v0.40.0/llm/llama_server.go#L2296-L2310
        if (think_set) {
            inputs.chat_template_kwargs["enable_thinking"] = enable_thinking ? "true" : "false";
            if (reasoning_effort != nullptr && reasoning_effort[0] != '\0') {
                inputs.chat_template_kwargs["reasoning_effort"] =
                    common_json::make(std::string(reasoning_effort)).dump();
            }
        }

        // Templates come from the model's GGUF (no overrides).
        auto tmpls = common_chat_templates_init(model, "");

        // Without a think kwarg, enable_thinking is derived from the template.
        // https://github.com/ggml-org/llama.cpp/blob/b11351/tools/server/server-context.cpp#L1454-L1465
        // https://github.com/ggml-org/llama.cpp/blob/b11351/tools/server/server-common.cpp#L1339-L1346
        inputs.enable_thinking = think_set
            ? (enable_thinking != 0)
            : common_chat_templates_support_enable_thinking(tmpls.get());

        auto params = common_chat_templates_apply(tmpls.get(), inputs);

        int n = (int) params.prompt.size();
        if (n >= buf_len) return -(n + 1);
        memcpy(buf, params.prompt.c_str(), n);
        buf[n] = '\0';
        return n;
    } catch (const std::exception & e) {
        if (buf_len > 0) { snprintf(buf, buf_len, "%s", e.what()); }
        return -1;
    }
}
