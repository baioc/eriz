const std = @import("std");

const root_file = "src/main.zig";

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const test_step = b.step("test", "Run library tests");
    {
        const tests = b.addTest(.{
            .root_source_file = .{ .path = root_file },
            .target = target,
            .optimize = optimize,
        });
        const run_tests = b.addRunArtifact(tests);
        test_step.dependOn(&run_tests.step);
    }

    const docs_step = b.step("docs", "Build documentation");
    {
        const emit_docs = b.addSystemCommand(&[_][]const u8{
            b.zig_exe,
            "test",
            root_file,
            "-femit-docs",
            "-fno-emit-bin",
        });
        docs_step.dependOn(&emit_docs.step);
    }

    b.default_step = test_step;
}
