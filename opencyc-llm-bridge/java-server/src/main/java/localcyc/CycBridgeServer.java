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

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.ByteArrayOutputStream;
import java.lang.reflect.Method;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.*;
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
 * - OpenCyc Java API jar provided by the user (see README)
 *
 * Endpoints are designed for a "Cyc tool" interface that an LLM-driven orchestrator can call.
 */
public class CycBridgeServer {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final int _HEALTH_CYC_TIMEOUT_MS = 5000;

   // private static final int _HEALTH_CYC_TIMEOUT_MS = 5000;

    private static class JsonHandler implements HttpHandler {
        private final CycBridge bridge;
        private final String route;

        JsonHandler(CycBridge bridge, String route) {
            this.bridge = bridge;
            this.route = route;
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

                switch (route) {
                    case "session":
                        resp = bridge.ensureSessionMt(body);
                        sendJson(exchange, 200, resp);
                        break;
                    case "constant_exists":
                        resp = bridge.constantExists(body);
                        sendJson(exchange, 200, resp);
                        break;
                    case "constant_create":
                        resp = bridge.createConstant(body);
                        sendJson(exchange, 200, resp);
                        break;
                    case "assert":
                        resp = bridge.assertSentence(body);
                        sendJson(exchange, 200, resp);
                        break;
                    case "ask_true":
                        resp = bridge.askTrue(body);
                        sendJson(exchange, 200, resp);
                        break;
                    case "ask_var":
                        resp = bridge.askVar(body);
                        sendJson(exchange, 200, resp);
                        break;
                    case "converse":
                        resp = bridge.converse(body);
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
        private final CycBridge bridge;

        HealthHandler(CycBridge bridge) {
            this.bridge = bridge;
        }

        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if (!"GET".equalsIgnoreCase(exchange.getRequestMethod())) {
                sendJson(exchange, 405, mapOf("ok", false, "error", "Method not allowed"));
                return;
            }

            Map<String, Object> info = new LinkedHashMap<>();
            info.put("ok", true);
            info.put("cyc_host", bridge.getCycHost());
            info.put("cyc_port", bridge.getCycPort());
            info.put("http_port", bridge.getHttpPort());
            info.put("timestamp_utc", new Date().toString());

            // sanity check: try fetching a known constant
            try {
                boolean canRead = callWithTimeout(new Callable<Boolean>() {
                    @Override
                    public Boolean call() throws Exception {
                        return bridge.canRead();
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

        server.setExecutor(null); // default executor
        System.out.println("CycBridgeServer listening on http://localhost:" + httpPort);
        System.out.println("Connecting to OpenCyc at " + cycHost + ":" + cycPort);

        server.start();
    }

    private static Map<String, Object> readJson(InputStream is) throws IOException {
        byte[] bytes = readAllBytesCompat(is);
        if (bytes.length == 0) return new HashMap<>();
        return MAPPER.readValue(bytes, new TypeReference<Map<String, Object>>() {});
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
        byte[] out = MAPPER.writeValueAsBytes(payload);
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
        private final CycAccess cyc;
        private final String cycHost;
        private final int cycPort;
        private final int httpPort;

        CycBridge(String host, int port, int httpPort) throws Exception {
            this.cycHost = host;
            this.cycPort = port;
            this.httpPort = httpPort;

            // The legacy OpenCyc API has different constructors across releases.
            // Try common ones via reflection.
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
            this.cyc = access;
        }

        public String getCycHost() { return cycHost; }
        public int getCycPort() { return cycPort; }
        public int getHttpPort() { return httpPort; }

        public boolean canRead() throws Exception {
            // Minimal connectivity check: these constants should exist in any OpenCyc KB.
            CycObject baseKb = cyc.getConstantByName("#$BaseKB");
            CycObject isa = cyc.getConstantByName("#$isa");
            return baseKb != null && isa != null;
        }

        // ---------- helpers ----------

        private static String normalizeConstantName(String name) {
            if (name == null) return null;
            name = name.trim();
            if (name.isEmpty()) return name;
            if (name.startsWith("#$")) return name;
            if (name.startsWith("?$")) return name; // not expected, but preserve
            if (name.startsWith("?")) return name;  // variable
            // allow user to pass "Dog" and treat as "#$Dog"
            return "#$" + name;
        }

        private static String stripConstantPrefix(String name) {
            if (name == null) return null;
            name = name.trim();
            if (name.startsWith("#$")) return name.substring(2);
            return name;
        }

        private static String sanitizeForConstantBareName(String bare) {
            // Cyc constant names are typically CamelCase with limited punctuation.
            // Keep alnum and underscore; replace others with underscore.
            return bare.replaceAll("[^A-Za-z0-9_]", "_");
        }

        private CycObject getMt(String mt) throws Exception {
            String mtName = normalizeConstantName(mt);
            try {
                return cyc.getConstantByName(mtName);
            } catch (CycApiException e) {
                throw new Exception("Unknown microtheory: " + mtName, e);
            }
        }

        // private CycList wrapIst(String mt, String sentenceOrQuery) throws Exception {
        //     String mtName = normalizeConstantName(mt);
        //     String s = "(#$ist " + mtName + " " + sentenceOrQuery + ")";
        //     return cyc.makeCycList(s);
        // }

        // private void assertInMt(String mt, String sentence) throws Exception {
        //     CycObject mtObj = getMt(mt);
        //     CycList assertion = wrapIst(mt, sentence);
        //     cyc.assertGaf(assertion, mtObj);
        // }
        private void assertInMt(String mt, String sentence) throws Exception {
            CycObject mtObj = getMt(mt);
            CycList assertion = cyc.makeCycList(sentence);
            cyc.assertGaf(assertion, mtObj);
        }

        // ---------- endpoints ----------

        public Map<String, Object> ensureSessionMt(Map<String, Object> body) throws Exception {
            String sessionId = (String) body.getOrDefault("session_id", UUID.randomUUID().toString().replace("-", ""));
            String comment = (String) body.getOrDefault("comment", "Auto-created session microtheory for Cyc LLM bridge.");
            String genl = (String) body.getOrDefault("genl_mt", "#$BaseKB");

            // Build a deterministic, readable constant name
            String bare = "CycLLMSessionMt_" + sanitizeForConstantBareName(sessionId);
            CycConstant mtConst = ensureConstantInternal(bare);

            String mtName = normalizeConstantName(mtConst.toString());
            // In some API versions, mtConst.toString() yields "#$Name"; in others it may yield "Name".
            // Ensure "#$".
            mtName = normalizeConstantName(stripConstantPrefix(mtName));

            // Ensure it's a microtheory and has a parent MT
            // These assertions go into BaseKB by default.
            // NOTE: You can change this to another administrative MT if desired.
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
                exists = cyc.getConstantByName(name) != null;
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
            // Try lookup first
            String full = normalizeConstantName(bareName);
            try {
                CycConstant existing = cyc.getConstantByName(full);
                if (existing != null) return existing;
            } catch (CycApiException ignored) {}

            // Try createNewPermanent(String)
            try {
                Method m = cyc.getClass().getMethod("createNewPermanent", String.class);
                Object o = m.invoke(cyc, bareName);
                if (o instanceof CycConstant) return (CycConstant) o;
            } catch (NoSuchMethodException ignored) {
                // fall through
            }

            // Try makeCycConstant(String) (some OpenCyc APIs)
            try {
                Method m = cyc.getClass().getMethod("makeCycConstant", String.class);
                Object o = m.invoke(cyc, bareName);
                if (o instanceof CycConstant) return (CycConstant) o;
            } catch (NoSuchMethodException ignored) {
                // fall through
            }

            // As a last resort, attempt to fetch again (some APIs create implicitly)
            try {
                CycConstant existing = cyc.getConstantByName(full);
                if (existing != null) return existing;
            } catch (CycApiException ignored) {}

            throw new Exception("Unable to create constant '" + bareName +
                    "'. Your OpenCyc Java API jar may not support createNewPermanent/makeCycConstant.");
        }

        public Map<String, Object> assertSentence(Map<String, Object> body) throws Exception {
            String mt = (String) body.getOrDefault("mt", "#$BaseKB");
            String sentence = (String) body.get("sentence");
            if (sentence == null) throw new Exception("Missing 'sentence'");

            // Defensive validation: the legacy OpenCyc API expects a CycList.
            // If the incoming sentence isn't a single parenthesized CycL form,
            // cyc.makeCycList(...) may throw a ClassCastException. Fail fast with a clear message.
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

            //CycObject mtObj = getMt(mt);
            //CycList q = wrapIst(mt, query);
            //boolean ans = cyc.isQueryTrue(q, mtObj);
            CycObject mtObj = getMt(mt);
            CycList q = cyc.makeCycList(query);
            boolean ans = cyc.isQueryTrue(q, mtObj);

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
                throw new Exception("CycL query must be fully parenthesized, e.g. '(#$isa #$Dog #$Animal)'. Got: " + qTrim);
            }

            // CycObject mtObj = getMt(mt);
            // CycList q = wrapIst(mt, query);
            // CycVariable v = CycObjectFactory.makeCycVariable(var);
            // CycList ret = cyc.askWithVariable(q, v, mtObj);
            CycObject mtObj = getMt(mt);
            CycList q = cyc.makeCycList(query);
            CycVariable v = CycObjectFactory.makeCycVariable(var);
            CycList ret = cyc.askWithVariable(q, v, mtObj);



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

            Map<String, Object> resp = new LinkedHashMap<>();
            resp.put("ok", true);
            resp.put("result", result);
            return resp;
        }

        private String converseInternal(String subl) throws Exception {
            // Prefer: cyc.converse().converseObject(subl)
            try {
                Method converseMethod = cyc.getClass().getMethod("converse");
                Object conn = converseMethod.invoke(cyc);
                Method convObjMethod = conn.getClass().getMethod("converseObject", String.class);
                Object result = convObjMethod.invoke(conn, subl);
                return result == null ? null : result.toString();
            } catch (NoSuchMethodException ignored) {
                // try next
            }

            // Fallback: cyc.converseObject(subl)
            try {
                Method m = cyc.getClass().getMethod("converseObject", String.class);
                Object result = m.invoke(cyc, subl);
                return result == null ? null : result.toString();
            } catch (NoSuchMethodException ignored) {
                // give up
            }

            throw new Exception("This OpenCyc Java API jar does not expose converse()/converseObject().");
        }

        private static String escapeForCycString(String s) {
            if (s == null) return "";
            // Cyc strings are in double quotes; escape backslash and quotes.
            return s.replace("\\", "\\\\").replace("\"", "\\\"");
        }
    }
}
