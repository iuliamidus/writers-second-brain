// A Writer's Second Brain — query book
//
// Paste any block into Neo4j Browser (http://localhost:7474). Queries that
// return nodes and relationships draw a picture; queries that return scalars
// give you a table. Both are useful, but the pictures are the point.
//
// Queries are ordered roughly by how much of the pipeline they require.
// Everything in sections 1-3 works with structural edges alone, so they run
// straight after `wsb graph push` with no LLM or embedding stage at all.


// ─────────────────────────────────────────────────────────────
// 1. Orientation — what is actually in here
// ─────────────────────────────────────────────────────────────

// Posts per blog, and the span of years each one covers.
MATCH (p:Post)-[:PUBLISHED_ON]->(b:Blog)
RETURN b.name AS blog,
       count(p) AS posts,
       min(p.year) AS from,
       max(p.year) AS until,
       sum(p.wordCount) AS words
ORDER BY posts DESC;


// Writing volume per year, across every blog at once. The shape of this is
// usually the first thing a writer has never seen before.
MATCH (p:Post)
WHERE p.year IS NOT NULL
RETURN p.year AS year, count(*) AS posts, sum(p.wordCount) AS words
ORDER BY year;


// Which languages, and how the mix shifted over time.
MATCH (p:Post) WHERE p.lang IS NOT NULL AND p.year IS NOT NULL
RETURN p.year AS year, p.lang AS lang, count(*) AS posts
ORDER BY year, posts DESC;


// ─────────────────────────────────────────────────────────────
// 2. Tags — the structure the writer built without noticing
// ─────────────────────────────────────────────────────────────

// The tag graph. Two tags are connected when they co-occur on a post, and
// the thicker the connection, the more habitual the pairing.
MATCH (t1:Tag)<-[:TAGGED]-(p:Post)-[:TAGGED]->(t2:Tag)
WHERE id(t1) < id(t2)
WITH t1, t2, count(p) AS shared
WHERE shared >= 3
RETURN t1, t2, shared
ORDER BY shared DESC
LIMIT 100;


// Tags that span the most years — long-running preoccupations rather than
// passing phases. These are chapter candidates.
MATCH (p:Post)-[:TAGGED]->(t:Tag)
WHERE p.year IS NOT NULL
WITH t, count(p) AS posts, min(p.year) AS first, max(p.year) AS last
WHERE posts >= 5
RETURN t.display AS tag, posts, first, last, last - first AS span
ORDER BY span DESC, posts DESC;


// Everything under one tag, oldest first — read a theme as a sequence.
MATCH (p:Post)-[:TAGGED]->(t:Tag {name: 'music'})
RETURN p.publishedAt AS date, p.title AS title, p.url AS url
ORDER BY date;


// ─────────────────────────────────────────────────────────────
// 3. Connections between posts
// ─────────────────────────────────────────────────────────────

// Series the writer numbered themselves, reassembled in order.
MATCH path = (a:Post)-[:CONTINUES*]->(b:Post)
WHERE NOT ()-[:CONTINUES]->(a)
RETURN path;


// Posts that link to each other — the writer's own cross-references,
// including the ones that jump between blogs.
MATCH (a:Post)-[l:LINKS_TO]->(b:Post)
MATCH (a)-[:PUBLISHED_ON]->(ba:Blog), (b)-[:PUBLISHED_ON]->(bb:Blog)
RETURN a, l, b, ba.name AS fromBlog, bb.name AS toBlog;


// Cross-blog neighbourhood: start from one post, walk two hops through
// shared tags, and surface what it connects to on a *different* blog.
// This is the query that finds the things fifteen years apart.
MATCH (seed:Post {id: $postId})-[:TAGGED]->(t:Tag)<-[:TAGGED]-(other:Post)
MATCH (seed)-[:PUBLISHED_ON]->(sb:Blog), (other)-[:PUBLISHED_ON]->(ob:Blog)
WHERE sb <> ob
WITH other, ob, count(DISTINCT t) AS sharedTags
RETURN other.title AS title, ob.name AS blog, other.publishedAt AS date, sharedTags
ORDER BY sharedTags DESC
LIMIT 25;


// Posts that drew the most reader response. Comments are a signal the writer
// already has and has probably never aggregated.
MATCH (p:Post)
WHERE p.commentCount > 0
RETURN p.title AS title, p.commentCount AS comments, p.year AS year, p.url AS url
ORDER BY comments DESC
LIMIT 25;


// ─────────────────────────────────────────────────────────────
// 4. Entities — needs the enrichment stage
// ─────────────────────────────────────────────────────────────

// The cultural references the writer keeps returning to: bands, films, books.
// For a writer who builds posts around songs, this is the spine of the work.
MATCH (p:Post)-[:MENTIONS]->(e:Entity)
WHERE e.kind = 'work'
WITH e, count(p) AS posts, min(p.year) AS first, max(p.year) AS last
WHERE posts >= 3
RETURN e.name AS work, posts, first, last, last - first AS span
ORDER BY span DESC, posts DESC;


// Every post that reached for one particular reference, over the years.
MATCH (p:Post)-[:MENTIONS]->(e:Entity {name: 'Pink Floyd'})
RETURN p.publishedAt AS date, p.title AS title, p.lang AS lang, p.url AS url
ORDER BY date;


// Entities that connect otherwise separate parts of the archive.
MATCH (a:Post)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(b:Post)
MATCH (a)-[:PUBLISHED_ON]->(ba:Blog), (b)-[:PUBLISHED_ON]->(bb:Blog)
WHERE ba <> bb
RETURN e.name AS entity, e.kind AS kind, count(DISTINCT a) + count(DISTINCT b) AS reach
ORDER BY reach DESC
LIMIT 30;


// ─────────────────────────────────────────────────────────────
// 5. Graph algorithms — needs the GDS plugin
// ─────────────────────────────────────────────────────────────

// Project a post-to-post graph, weighted by however many ways two posts are
// connected. Run this once per session before the algorithms below.
CALL gds.graph.project.cypher(
  'posts',
  'MATCH (p:Post) RETURN id(p) AS id',
  'MATCH (a:Post)-[r]-(b:Post) RETURN id(a) AS source, id(b) AS target,
   coalesce(r.weight, 1.0) AS weight',
  {validateRelationships: false}
) YIELD graphName, nodeCount, relationshipCount;


// Leiden communities — the clustering that produces chapter candidates.
CALL gds.leiden.stream('posts', {relationshipWeightProperty: 'weight'})
YIELD nodeId, communityId
WITH gds.util.asNode(nodeId) AS p, communityId
RETURN communityId,
       count(*) AS posts,
       min(p.year) AS first,
       max(p.year) AS last,
       collect(p.title)[0..8] AS sample
ORDER BY posts DESC;


// Write communities back so you can colour by them in Browser.
CALL gds.leiden.write('posts', {
  relationshipWeightProperty: 'weight',
  writeProperty: 'community'
}) YIELD communityCount, modularity;


// Betweenness centrality — bridge posts. High scores sit between clusters,
// which makes them the natural transitions in a longer piece of writing.
CALL gds.betweenness.stream('posts')
YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS p, score
WHERE score > 0
RETURN p.title AS title, p.year AS year, round(score) AS betweenness, p.url AS url
ORDER BY betweenness DESC
LIMIT 25;


// Drop the projection when you are done.
CALL gds.graph.drop('posts') YIELD graphName;
