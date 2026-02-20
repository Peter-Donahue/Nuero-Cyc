package localcyc;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import org.opencyc.api.CycAccess;
import org.opencyc.api.CycApiException;
import org.opencyc.api.CycObjectFactory;
import org.opencyc.cycobject.CycConstant;
import org.opencyc.cycobject.CycList;
import org.opencyc.cycobject.CycObject;
import org.opencyc.cycobject.CycVariable;
import org.opencyc.cycobject.DefaultCycObject;
import org.opencyc.parser.CycLParserUtil;
import org.opencyc.parser.ParseException;
import org.opencyc.parser.TokenMgrError;
import org.opencyc.parser.UnsupportedVocabularyException;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.reflect.Method;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/**
 * Minimal local HTTP wrapper around the legacy OpenCyc Java API.
 *
 * This server is intentionally small and dependency-light:
 * - Java built-in HttpServer
 * - Jackson for JSON
 * - OpenCyc Java API jar provided by the user
 *
 * Endpoints are designed for a "Cyc tool" interface that a Python CLI can call.
 */
public class CycBridgeServer {

    private static final ObjectMapper _MAPPER = new ObjectMapper();
    private static final int _HEALTH_CYC_TIMEOUT_MS = 5000;

    private static class JsonHandler implements HttpHandler {
        private final CycBridge _BRIDGE;
        private final String _ROUTE;

        JsonHandler(CycBridge bridge, String route) {
            this._BRIDGE = bridge;
            this._ROUTE = route;
        }

        @Override
        public void handle(HttpExchange exchange) throws IOException {
            try {
                if (!"POST".equalsIgnoreCase(exchange.getRequestMethod())) {
                    sendJson(exchange, 405, mapOf("ok", false, "error", "Method not allowed"));
                    return;
                }

                Map<String, Object> body = readJson(exchange.getRequestBody());
                Map<String, Object> resp;

                switch (_ROUTE) {
                    case "session":
                        resp = _BRIDGE.ensureSessionMt(body);
                        sendJson(exchange, 200, resp);
                        break;
                    case "constant_exists":
                        resp = _BRIDGE.constantExists(body);
                        sendJson(exchange, 200, resp);
                        break;
                    case "constant_create":
                        resp = _BRIDGE.createConstant(body);
                        sendJson(exchange, 200, resp);
                        break;
                    case "assert":
                        resp = _BRIDGE.assertSentence(body);
                        sendJson(exchange, 200, resp);
                        break;
                    case "ask_true":
                        resp = _BRIDGE.askTrue(body);
                        sendJson(exchange, 200, resp);
                        break;
                    case "ask_var":
                        resp = _BRIDGE.askVar(body);
                        sendJson(exchange, 200, resp);
                        break;
                    case "converse":
                        resp = _BRIDGE.converse(body);
                        sendJson(exchange, 200, resp);
                        break;
                    case "lex_lookup":
                        resp = _BRIDGE.lexLookup(body);
                        sendJson(exchange, 200, resp);
                        break;
                    case "cycl_parse":
                        resp = _BRIDGE.cyclParse(body);
                        sendJson(exchange, 200, resp);
                        break;
                    default:
                        sendJson(exchange, 404, mapOf("ok", false, "error", "Unknown route"));
                }
            } catch (Throwable t) {
                sendJson(exchange, 500, mapOf(
                        "ok", false,
                        "error", t.getClass().getSimpleName(),
                        "message", t.getMessage()
                ));
            }
        }
    }

    private static class HealthHandler implements HttpHandler {
        private final CycBridge _BRIDGE;

        HealthHandler(CycBridge bridge) {
            this._BRIDGE = bridge;
        }

        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                sendJson(exchange, 405, mapOf("ok", false, "error", "Method not allowed"));
                return;
            }

            Map<String, Object> info = new LinkedHashMap<>();
            info.put("ok", true);
            info.put("cyc_host", _BRIDGE.getCycHost());
            info.put("cyc_port", _BRIDGE.getCycPort());
            info.put("http_port", _BRIDGE.getHttpPort());
            info.put("timestamp_utc", new Date().toString());

            try {
                boolean canRead = callWithTimeout(new Callable<Boolean>() {
                    @Override
                    public Boolean call() throws Exception {
                        return _BRIDGE.canRead();
                    }
                }, _HEALTH_CYC_TIMEOUT_MS);
                info.put("cyc_connected", canRead);
            } catch (Throwable t) {
                info.put("cyc_connected", false);
                info.put("cyc_error", t.getMessage());
            }

            sendJson(exchange, 200, info);
        }
    }

    public static void main(String[] args) throws Exception {
        String cycHost = getenv("CYC_HOST", "localhost");
        int cycPort = Integer.parseInt(getenv("CYC_PORT", "3601"));
        int httpPort = Integer.parseInt(getenv("CYC_BRIDGE_HTTP_PORT", "8081"));

        CycBridge bridge = new CycBridge(cycHost, cycPort, httpPort);

        HttpServer server = HttpServer.create(new InetSocketAddress(httpPort), 0);

        server.createContext("/health", new HealthHandler(bridge));

        server.createContext("/api/v1/session", new JsonHandler(bridge, "session"));
        server.createContext("/api/v1/constant/exists", new JsonHandler(bridge, "constant_exists"));
        server.createContext("/api/v1/constant/create", new JsonHandler(bridge, "constant_create"));
        server.createContext("/api/v1/assert", new JsonHandler(bridge, "assert"));
        server.createContext("/api/v1/ask_true", new JsonHandler(bridge, "ask_true"));
        server.createContext("/api/v1/ask_var", new JsonHandler(bridge, "ask_var"));
        server.createContext("/api/v1/converse", new JsonHandler(bridge, "converse"));

        server.createContext("/api/v1/lex/lookup", new JsonHandler(bridge, "lex_lookup"));
        server.createContext("/api/v1/cycl/parse", new JsonHandler(bridge, "cycl_parse"));

        server.setExecutor(null); // default executor
        System.out.println("CycBridgeServer listening on http://localhost:" + httpPort);
        System.out.println("Connecting to OpenCyc at " + cycHost + ":" + cycPort);

        server.start();
    }

    private static Map<String, Object> readJson(InputStream is) throws IOException {
        byte[] bytes = readAllBytesCompat(is);
        if (bytes.length == 0) return new HashMap<>();
        return _MAPPER.readValue(bytes, new TypeReference<Map<String, Object>>() {});
    }

    private static byte[] readAllBytesCompat(InputStream is) throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        byte[] buf = new byte[8192];
        int n;
        while ((n = is.read(buf)) != -1) {
            bos.write(buf, 0, n);
        }
        return bos.toByteArray();
    }

    private static Map<String, Object> mapOf(Object... keyVals) {
        if (keyVals == null || keyVals.length == 0) return new LinkedHashMap<>();
        if ((keyVals.length % 2) != 0) throw new IllegalArgumentException("mapOf requires even number of args");
        Map<String, Object> m = new LinkedHashMap<>();
        for (int i = 0; i < keyVals.length; i += 2) {
            String k = String.valueOf(keyVals[i]);
            Object v = keyVals[i + 1];
            m.put(k, v);
        }
        return m;
    }

    private static <T> T callWithTimeout(Callable<T> call, int timeoutMs) throws Exception {
        ExecutorService exec = Executors.newSingleThreadExecutor();
        Future<T> fut = exec.submit(call);
        try {
            return fut.get(timeoutMs, TimeUnit.MILLISECONDS);
        } catch (TimeoutException te) {
            fut.cancel(true);
            throw new Exception("Timed out after " + timeoutMs + "ms", te);
        } finally {
            exec.shutdownNow();
        }
    }

    private static void sendJson(HttpExchange exchange, int status, Map<String, Object> payload) throws IOException {
        byte[] out = _MAPPER.writeValueAsBytes(payload);
        Headers headers = exchange.getResponseHeaders();
        headers.set("Content-Type", "application/json; charset=utf-8");
        headers.set("Access-Control-Allow-Origin", "*");
        exchange.sendResponseHeaders(status, out.length);
        try (OutputStream os = exchange.getResponseBody()) {
            os.write(out);
        }
    }

    private static String getenv(String key, String def) {
        String v = System.getenv(key);
        if (v == null || v.isEmpty()) return def;
        return v;
    }

    /**
     * Wrapper around OpenCyc Java API calls.
     * Keeps the OpenCyc connection alive and provides safe-ish endpoints.
     */
    static class CycBridge {
        private final CycAccess _CYC;
        private final String _CYC_HOST;
        private final int _CYC_PORT;
        private final int _HTTP_PORT;

        CycBridge(String host, int port, int httpPort) throws Exception {
            this._CYC_HOST = host;
            this._CYC_PORT = port;
            this._HTTP_PORT = httpPort;

            CycAccess access = null;
            Exception last = null;
            try {
                access = new CycAccess(host, port);
            } catch (Throwable t) {
                // ignore; try reflection fallbacks
            }
            if (access == null) {
                try {
                    access = new CycAccess();
                } catch (Throwable t) {
                    last = new Exception("Failed to initialize CycAccess (no-arg constructor)", t);
                }
            }
            if (access == null) throw last != null ? last : new Exception("Failed to initialize CycAccess");
            this._CYC = access;
        }

        public String getCycHost() { return _CYC_HOST; }
        public int getCycPort() { return _CYC_PORT; }
        public int getHttpPort() { return _HTTP_PORT; }

        public boolean canRead() throws Exception {
            CycObject baseKb = _CYC.getConstantByName("#$BaseKB");
            CycObject isa = _CYC.getConstantByName("#$isa");
            return baseKb != null && isa != null;
        }

        // ---------- helpers ----------

        private static String normalizeConstantName(String name) {
            if (name == null) return null;
            name = name.trim();
            if (name.isEmpty()) return name;
            if (name.startsWith("#$")) return name;
            if (name.startsWith("?$")) return name;
            if (name.startsWith("?")) return name;  // variable
            if (name.startsWith("(")) return name;  // NAUT / formula
            if (name.startsWith("\"")) return name; // string literal
            return "#$" + name;
        }

        private static String stripConstantPrefix(String name) {
            if (name == null) return null;
            name = name.trim();
            if (name.startsWith("#$")) return name.substring(2);
            return name;
        }

        private static String sanitizeForConstantBareName(String bare) {
            return bare.replaceAll("[^A-Za-z0-9_]", "_");
        }

        private CycObject getMt(String mt) throws Exception {
            String mtName = normalizeConstantName(mt);
            try {
                return _CYC.getConstantByName(mtName);
            } catch (CycApiException e) {
                throw new Exception("Unknown microtheory: " + mtName, e);
            }
        }

        private void assertInMt(String mt, String sentence) throws Exception {
            CycObject mtObj = getMt(mt);
            CycList assertion = _CYC.makeCycList(sentence);
            _CYC.assertGaf(assertion, mtObj);
        }

        private static String escapeForCycString(String s) {
            if (s == null) return "";
            return s.replace("\\", "\\\\").replace("\"", "\\\"");
        }

        private static String safeCycToString(Object obj) {
            if (obj == null) return null;
            try {
                return DefaultCycObject.cyclifyWithEscapeChars(obj, true);
            } catch (Throwable t) {
                return obj.toString();
            }
        }

        // ---------- endpoints ----------

        public Map<String, Object> ensureSessionMt(Map<String, Object> body) throws Exception {
            String sessionId = (String) body.getOrDefault("session_id", UUID.randomUUID().toString().replace("-", ""));
            String comment = (String) body.getOrDefault("comment", "Auto-created session microtheory for Cyc bridge.");
            String genl = (String) body.getOrDefault("genl_mt", "#$BaseKB");

            String bare = "CycLLMSessionMt_" + sanitizeForConstantBareName(sessionId);
            CycConstant mtConst = ensureConstantInternal(bare);

            String mtName = normalizeConstantName(stripConstantPrefix(mtConst.toString()));

            String adminMt = "#$BaseKB";
            assertInMt(adminMt, "(#$isa " + mtName + " #$Microtheory)");
            assertInMt(adminMt, "(#$genlMt " + mtName + " " + normalizeConstantName(genl) + ")");
            assertInMt(adminMt, "(#$comment " + mtName + " \"" + escapeForCycString(comment) + "\")");

            Map<String, Object> resp = new LinkedHashMap<>();
            resp.put("ok", true);
            resp.put("session_id", sessionId);
            resp.put("session_mt", mtName);
            resp.put("genl_mt", normalizeConstantName(genl));
            return resp;
        }

        public Map<String, Object> constantExists(Map<String, Object> body) throws Exception {
            String nameRaw = (String) body.get("name");
            if (nameRaw == null) throw new Exception("Missing 'name'");
            String name = normalizeConstantName(nameRaw);

            boolean exists;
            try {
                exists = _CYC.getConstantByName(name) != null;
            } catch (CycApiException e) {
                exists = false;
            }

            return mapOf("ok", true, "name", name, "exists", exists);
        }

        public Map<String, Object> createConstant(Map<String, Object> body) throws Exception {
            String nameRaw = (String) body.get("name");
            if (nameRaw == null) throw new Exception("Missing 'name'");
            String bare = sanitizeForConstantBareName(stripConstantPrefix(nameRaw));
            CycConstant c = ensureConstantInternal(bare);
            String name = normalizeConstantName(stripConstantPrefix(c.toString()));
            return mapOf("ok", true, "name", name, "created", true);
        }

        private CycConstant ensureConstantInternal(String bareName) throws Exception {
            String full = normalizeConstantName(bareName);
            try {
                CycConstant existing = _CYC.getConstantByName(full);
                if (existing != null) return existing;
            } catch (CycApiException ignored) {}

            try {
                Method m = _CYC.getClass().getMethod("createNewPermanent", String.class);
                Object o = m.invoke(_CYC, bareName);
                if (o instanceof CycConstant) return (CycConstant) o;
            } catch (NoSuchMethodException ignored) {
                // fall through
            }

            try {
                Method m = _CYC.getClass().getMethod("makeCycConstant", String.class);
                Object o = m.invoke(_CYC, bareName);
                if (o instanceof CycConstant) return (CycConstant) o;
            } catch (NoSuchMethodException ignored) {
                // fall through
            }

            try {
                CycConstant existing = _CYC.getConstantByName(full);
                if (existing != null) return existing;
            } catch (CycApiException ignored) {}

            throw new Exception("Unable to create constant '" + bareName +
                    "'. Your OpenCyc Java API jar may not support createNewPermanent/makeCycConstant.");
        }

        public Map<String, Object> assertSentence(Map<String, Object> body) throws Exception {
            String mt = (String) body.getOrDefault("mt", "#$BaseKB");
            String sentence = (String) body.get("sentence");
            if (sentence == null) throw new Exception("Missing 'sentence'");

            String sTrim = sentence.trim();
            if (!(sTrim.startsWith("(") && sTrim.endsWith(")"))) {
                throw new Exception("CycL sentence must be fully parenthesized, e.g. '(#$isa #$Dog #$Animal)'. Got: " + sTrim);
            }

            assertInMt(mt, sentence);
            return mapOf("ok", true);
        }

        public Map<String, Object> askTrue(Map<String, Object> body) throws Exception {
            String mt = (String) body.getOrDefault("mt", "#$BaseKB");
            String query = (String) body.get("query");
            if (query == null) throw new Exception("Missing 'query'");

            String qTrim = query.trim();
            if (!(qTrim.startsWith("(") && qTrim.endsWith(")"))) {
                throw new Exception("CycL query must be fully parenthesized, e.g. '(#$isa #$Dog #$Animal)'. Got: " + qTrim);
            }

            CycObject mtObj = getMt(mt);
            CycList q = _CYC.makeCycList(query);
            boolean ans = _CYC.isQueryTrue(q, mtObj);

            return mapOf("ok", true, "answer", ans);
        }

        public Map<String, Object> askVar(Map<String, Object> body) throws Exception {
            String mt = (String) body.getOrDefault("mt", "#$BaseKB");
            String query = (String) body.get("query");
            String var = (String) body.getOrDefault("var", "?X");
            int limit = ((Number) body.getOrDefault("limit", 50)).intValue();

            if (query == null) throw new Exception("Missing 'query'");

            String qTrim = query.trim();
            if (!(qTrim.startsWith("(") && qTrim.endsWith(")"))) {
                throw new Exception("CycL query must be fully parenthesized, e.g. '(#$isa ?X #$Dog)'. Got: " + qTrim);
            }

            CycObject mtObj = getMt(mt);
            CycList q = _CYC.makeCycList(query);
            CycVariable v = CycObjectFactory.makeCycVariable(var);
            CycList ret = _CYC.askWithVariable(q, v, mtObj);

            List<String> bindings = new ArrayList<>();
            int n = Math.min(limit, ret.size());
            for (int i = 0; i < n; i++) {
                Object o = ret.get(i);
                bindings.add(o == null ? null : o.toString());
            }

            Map<String, Object> resp = new LinkedHashMap<>();
            resp.put("ok", true);
            resp.put("bindings", bindings);
            resp.put("count", bindings.size());
            return resp;
        }

        public Map<String, Object> converse(Map<String, Object> body) throws Exception {
            String subl = (String) body.get("subl");
            if (subl == null) throw new Exception("Missing 'subl'");

            String result = converseInternal(subl);
            return mapOf("ok", true, "result", result);
        }

        private String converseInternal(String subl) throws Exception {
            try {
                Method converseMethod = _CYC.getClass().getMethod("converse");
                Object conn = converseMethod.invoke(_CYC);
                Method convObjMethod = conn.getClass().getMethod("converseObject", String.class);
                Object result = convObjMethod.invoke(conn, subl);
                return result == null ? null : result.toString();
            } catch (NoSuchMethodException ignored) {
                // try next
            }

            try {
                Method m = _CYC.getClass().getMethod("converseObject", String.class);
                Object result = m.invoke(_CYC, subl);
                return result == null ? null : result.toString();
            } catch (NoSuchMethodException ignored) {
                // give up
            }

            throw new Exception("This OpenCyc Java API jar does not expose converse()/converseObject().");
        }

        // ----------------------------
        // Lexicon lookup endpoint
        // ----------------------------

        public Map<String, Object> lexLookup(Map<String, Object> body) throws Exception {
            String kind = (String) body.getOrDefault("kind", "noun");
            String lemma = (String) body.get("lemma");
            String text = (String) body.get("text");
            String mt = (String) body.getOrDefault("mt", "#$EnglishLexicalMt");
            int limit = ((Number) body.getOrDefault("limit", 50)).intValue();

            String key = (lemma != null && !lemma.isEmpty()) ? lemma : text;
            if (key == null || key.trim().isEmpty()) {
                throw new Exception("lex_lookup requires 'lemma' or 'text'");
            }

            String escaped = escapeForCycString(key.trim());

            String query;
            String var;
            if ("noun".equalsIgnoreCase(kind)) {
                query = "(#$and (#$lex ?FORM ?W (\"" + escaped + "\")) (#$denotation ?W #$SimpleNoun 0 ?DENOT))";
                var = "?DENOT";
            } else if ("verb".equalsIgnoreCase(kind)) {
                query = "(#$and (#$lex ?FORM ?W (\"" + escaped + "\")) (#$denotation ?W #$Verb 0 ?PRED))";
                var = "?PRED";
            } else if ("adj".equalsIgnoreCase(kind) || "adjective".equalsIgnoreCase(kind)) {
                query = "(#$and (#$lex ?FORM ?W (\"" + escaped + "\")) (#$denotation ?W #$Adjective 0 ?PRED))";
                var = "?PRED";
            } else if ("proper".equalsIgnoreCase(kind) || "name".equalsIgnoreCase(kind)) {
                // names are usually in BaseKB or a general mt; caller can override mt.
                query = "(#$and (#$nameString ?C \"" + escaped + "\") (#$isa ?C #$Individual))";
                var = "?C";
            } else {
                throw new Exception("Unknown lex_lookup kind: " + kind);
            }

            CycObject mtObj = getMt(mt);
            CycList q = _CYC.makeCycList(query);
            CycVariable v = CycObjectFactory.makeCycVariable(var);
            CycList ret = _CYC.askWithVariable(q, v, mtObj);

            List<String> candidates = new ArrayList<>();
            int n = Math.min(limit, ret.size());
            for (int i = 0; i < n; i++) {
                Object o = ret.get(i);
                candidates.add(o == null ? null : o.toString());
            }

            Map<String, Object> resp = new LinkedHashMap<>();
            resp.put("ok", true);
            resp.put("kind", kind);
            resp.put("key", key);
            resp.put("mt", normalizeConstantName(mt));
            resp.put("query", query);
            resp.put("var", var);
            resp.put("candidates", candidates);
            resp.put("count", candidates.size());
            return resp;
        }

        // ----------------------------
        // CycL parsing/canonicalization endpoint
        // ----------------------------

        public Map<String, Object> cyclParse(Map<String, Object> body) throws Exception {
            String text = (String) body.get("text");
            if (text == null) throw new Exception("Missing 'text'");

            String kind = (String) body.getOrDefault("kind", "sentence");
            boolean testEof = (body.get("test_eof") instanceof Boolean) ? (Boolean) body.get("test_eof") : true;
            String mode = (String) body.getOrDefault("mode", "none");

            Object parsed;
            try {
                if ("sentence".equalsIgnoreCase(kind)) {
                    parsed = CycLParserUtil.parseCycLSentence(text, testEof, _CYC);
                } else if ("term".equalsIgnoreCase(kind)) {
                    parsed = CycLParserUtil.parseCycLTerm(text, testEof, _CYC);
                } else if ("term_list".equalsIgnoreCase(kind)) {
                    parsed = CycLParserUtil.parseCycLTermList(text, testEof, _CYC);
                } else if ("string".equalsIgnoreCase(kind)) {
                    parsed = CycLParserUtil.parseCycLString(text, testEof, _CYC);
                } else if ("number".equalsIgnoreCase(kind)) {
                    parsed = CycLParserUtil.parseCycLNumber(text, testEof, _CYC);
                } else if ("constant".equalsIgnoreCase(kind)) {
                    parsed = CycLParserUtil.parseCycLConstant(text, testEof, _CYC);
                } else if ("variable".equalsIgnoreCase(kind)) {
                    parsed = CycLParserUtil.parseCycLVariable(text, testEof, _CYC);
                } else {
                    throw new Exception("Unknown kind: " + kind);
                }

                if ("nart".equalsIgnoreCase(mode) || "nart_substitute".equalsIgnoreCase(mode)) {
                    parsed = CycLParserUtil.nartSubstitute(parsed, _CYC);
                } else if ("hl".equalsIgnoreCase(mode) || "to_hl".equalsIgnoreCase(mode)) {
                    parsed = CycLParserUtil.toHL(parsed, _CYC);
                }

            } catch (ParseException | UnsupportedVocabularyException | TokenMgrError e) {
                Map<String, Object> err = new LinkedHashMap<>();
                err.put("ok", false);
                err.put("error", e.getClass().getSimpleName());
                err.put("message", e.getMessage());
                err.put("kind", kind);
                return err;
            }

            Map<String, Object> resp = new LinkedHashMap<>();
            resp.put("ok", true);
            resp.put("kind", kind);
            resp.put("mode", mode);
            resp.put("java_class", parsed == null ? null : parsed.getClass().getName());
            resp.put("value", parsed == null ? null : parsed.toString());
            resp.put("cyclified", safeCycToString(parsed));
            return resp;
        }
    }
}
