#include "priority.cuh"
#include <cstdio>
#include <cassert>

static int passed = 0;
static int failed = 0;

#define TEST(name) \
    printf("  TEST: %-50s ", #name); \
    test_##name(); \
    printf("PASS\n"); passed++;

void test_basic_priority() {
    uint64_t p = compute_priority(10, 42);
    assert(priority_degree(p) == 10);
    assert(priority_vertex_id(p) == 42);
}

void test_higher_degree_wins() {
    uint64_t p1 = compute_priority(100, 1);  // degree 100, id 1
    uint64_t p2 = compute_priority(50, 999); // degree 50, id 999
    assert(priority_wins(p1, p2));            // higher degree wins
    assert(!priority_wins(p2, p1));
}

void test_same_degree_higher_id_wins() {
    uint64_t p1 = compute_priority(100, 500);
    uint64_t p2 = compute_priority(100, 200);
    assert(priority_wins(p1, p2));            // same degree, higher id wins
    assert(!priority_wins(p2, p1));
}

void test_uniqueness() {
    // No two different vertices can have the same priority
    uint64_t p1 = compute_priority(10, 1);
    uint64_t p2 = compute_priority(10, 2);
    assert(p1 != p2);

    uint64_t p3 = compute_priority(11, 1);
    assert(p1 != p3);
}

void test_zero_degree() {
    uint64_t p = compute_priority(0, 42);
    assert(priority_degree(p) == 0);
    assert(priority_vertex_id(p) == 42);

    // Zero-degree vertex loses to any non-zero degree
    uint64_t p2 = compute_priority(1, 0);
    assert(priority_wins(p2, p));
}

void test_max_values() {
    uint64_t p = compute_priority(0xFFFFFFFF, 0xFFFFFFFF);
    assert(priority_degree(p) == 0xFFFFFFFF);
    assert(priority_vertex_id(p) == 0xFFFFFFFF);
}

void test_deterministic_winner() {
    // For any pair of distinct vertices, exactly one wins
    uint64_t p1 = compute_priority(50, 100);
    uint64_t p2 = compute_priority(50, 200);
    assert(priority_wins(p1, p2) != priority_wins(p2, p1));  // exactly one wins
}

void test_transitivity() {
    uint64_t p1 = compute_priority(100, 3);
    uint64_t p2 = compute_priority(50, 999);
    uint64_t p3 = compute_priority(25, 5000);
    assert(priority_wins(p1, p2));
    assert(priority_wins(p2, p3));
    assert(priority_wins(p1, p3));  // transitive
}

int main() {
    printf("=== Priority Unit Tests ===\n");

    TEST(basic_priority);
    TEST(higher_degree_wins);
    TEST(same_degree_higher_id_wins);
    TEST(uniqueness);
    TEST(zero_degree);
    TEST(max_values);
    TEST(deterministic_winner);
    TEST(transitivity);

    printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
