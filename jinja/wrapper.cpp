#include "parser.h"
#include "runtime.h"
#include "value.h"
#include "caps.h"

#include <cstring>
#include <string>
#include <vector>

using namespace jinja;

// Mirrors chat.cpp:2192-2210,2700 — if the template doesn't support system
// role, merge system content into the next message.
static void merge_system_if_unsupported(
    program & prog,
    std::vector<std::string> & roles,
    std::vector<std::string> & contents)
{
    if (caps_get(prog).supports_system_role) return;
    if (roles.empty() || roles[0] != "system") return;
    if (roles.size() > 1) {
        contents[1] = contents[0] + "\n" + contents[1];
        roles.erase(roles.begin());
        contents.erase(contents.begin());
    } else {
        roles.erase(roles.begin());
        contents.erase(contents.begin());
    }
}

struct ot_jinja_message {
    const char * role;
    const char * content;
};

// https://github.com/ggml-org/llama.cpp/blob/b9888/common/chat.cpp#L2192-L2210
extern "C" int ot_jinja_render(
    const char * tmpl,
    const ot_jinja_message * msgs_in, int n_msgs,
    const char * bos_token,
    const char * eos_token,
    int add_generation_prompt,
    char * buf, int buf_len)
{
    try {
        lexer lex;
        auto lex_res = lex.tokenize(tmpl);
        auto prog = parse_from_tokens(lex_res);

        std::vector<std::string> roles(n_msgs);
        std::vector<std::string> contents(n_msgs);
        for (int i = 0; i < n_msgs; i++) {
            roles[i]    = msgs_in[i].role;
            contents[i] = msgs_in[i].content;
        }
        merge_system_if_unsupported(prog, roles, contents);
        n_msgs = (int) roles.size();

        auto arr = mk_val<value_array_t>();
        for (int i = 0; i < n_msgs; i++) {
            auto obj = mk_val<value_object_t>();
            obj->insert("role", mk_val<value_string_t>(roles[i]));
            auto content = mk_val<value_string_t>(contents[i]);
            content->mark_input();
            obj->insert("content", content);
            arr->val_arr.push_back(obj);
        }

        context ctx(lex_res.source);
        ctx.set_val("messages",              arr);
        ctx.set_val("add_generation_prompt", mk_val<value_bool_t>((bool) add_generation_prompt));
        ctx.set_val("bos_token",             mk_val<value_string_t>(bos_token ? bos_token : ""));
        ctx.set_val("eos_token",             mk_val<value_string_t>(eos_token ? eos_token : ""));

        runtime rt(ctx);
        auto results = rt.execute(prog);
        auto parts   = runtime::gather_string_parts(results);

        std::string output;
        for (const auto & p : parts->val_str.parts) {
            output += p.val;
        }

        int n = (int) output.size();
        if (n >= buf_len) return -(n + 1);
        memcpy(buf, output.c_str(), n);
        buf[n] = '\0';
        return n;
    } catch (const std::exception &) {
        if (buf_len > 0) buf[0] = '\0';
        return -1;
    }
}
